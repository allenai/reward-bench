"""Offline text-template regressions; runnable with python tests/test_conversation.py.

The golden hashes were generated from the hash-verified fschat 0.2.36 wheel,
plus RewardBench's three original custom template registrations. They cover
configuration, prompts, API messages and mutation across every legacy template.
Import the module by path so these checks need neither FastChat nor ML packages.
"""

import ast
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


conversation = load_module(ROOT / "rewardbench" / "conversation.py", "rewardbench_conversation_test")


def result_or_error(callback):
    try:
        return {"value": callback()}
    except ValueError as error:
        # Several upstream API-only templates deliberately have no prompt style.
        return {"error": type(error).__name__, "message": str(error)}


def observe(conv):
    return {
        "prompt": result_or_error(conv.get_prompt),
        "api_messages": conv.to_openai_api_messages(),
        "conversation": conv.dict(),
    }


def template_fingerprints(module):
    result = {}
    for name in sorted(module.conv_templates):
        conv = module.get_conv_template(name)
        observations = [module.dataclasses.asdict(conv), observe(conv)]
        conv.append_message(conv.roles[0], "What is 2 + 2? café ☃\r\nline\n\nend")
        conv.append_message(conv.roles[1], None)
        observations.append(observe(conv))
        conv.update_last_message("Four.\nAn answer.")
        observations.append(observe(conv))
        conv.append_message(conv.roles[0], "")
        conv.append_message(conv.roles[1], "")
        observations.append(observe(conv))
        for system in ["System override.\nTwo lines.", ""]:
            conv = module.get_conv_template(name)
            conv.set_system_message(system)
            conv.messages = []
            for index, message in enumerate(["Hello", "Hi", "Another question", None]):
                conv.append_message(conv.roles[index % 2], message)
            observations.append(observe(conv))
        payload = json.dumps(observations, sort_keys=True, ensure_ascii=False).encode()
        result[name] = hashlib.sha256(payload).hexdigest()
    return result


class ConversationTest(unittest.TestCase):
    def test_all_legacy_templates_match_fastchat_reference(self):
        fixture = json.loads((ROOT / "tests/fixtures/fastchat_text_templates.json").read_text())
        actual = template_fingerprints(conversation)
        self.assertEqual(set(actual), set(fixture["templates"]))
        for name, expected in fixture["templates"].items():
            with self.subTest(template=name):
                self.assertEqual(actual[name], expected)

    def test_rewardbench_prompt_formats(self):
        expected = {
            "tulu": "<|user|>\nHello\n<|assistant|>\nHi\n",
            "raw": "HelloHi",
            "pku-align": "BEGINNING OF CONVERSATION: USER: Hello ASSISTANT: Hi ",
            "openbmb": "User: Hello\n\nAssistant: Hi\n\n",
            "Ziya": "\n\nHuman: Hello\n\nAssistant: Hi\n\n",
            "oasst_pythia": "<|prompter|>Hello<|endoftext|><|assistant|>Hi<|endoftext|>",
        }
        for name, prompt in expected.items():
            with self.subTest(template=name):
                conv = conversation.get_conv_template(name)
                conv.append_message(conv.roles[0], "Hello")
                conv.append_message(conv.roles[1], "Hi")
                self.assertEqual(conv.get_prompt(), prompt)

    def test_api_message_format(self):
        for name in ["chatgpt", "claude"]:
            with self.subTest(template=name):
                conv = conversation.get_conv_template(name)
                conv.set_system_message("Judge the responses.")
                conv.append_message(conv.roles[0], "Which is better?")
                conv.append_message(conv.roles[1], None)
                self.assertEqual(
                    conv.to_openai_api_messages(),
                    [
                        {"role": "system", "content": "Judge the responses."},
                        {"role": "user", "content": "Which is better?"},
                    ],
                )

    def test_template_messages_are_independent(self):
        first = conversation.get_conv_template("one_shot")
        second = conversation.get_conv_template("one_shot")
        first.update_last_message("A changed answer")
        first.append_message(first.roles[0], "A new message")
        self.assertNotEqual(first.messages, second.messages)
        self.assertEqual(second.messages, conversation.get_conv_template("one_shot").messages)

    def test_image_messages_are_rejected(self):
        conv = conversation.get_conv_template("vicuna_v1.1")
        image_message = ("Describe this", ["http://127.0.0.1/private.png"])
        with self.assertRaisesRegex(TypeError, "text messages only"):
            conv.append_message(conv.roles[0], image_message)
        # Existing callers also assign messages directly.
        conv.messages = [[conv.roles[0], image_message]]
        for callback in [conv.get_prompt, conv.to_openai_api_messages, conv.dict, conv.copy]:
            with self.subTest(method=callback.__name__):
                with self.assertRaisesRegex(TypeError, "text messages only"):
                    callback()

    def test_adapter_has_no_external_or_image_helpers(self):
        source = (ROOT / "rewardbench/conversation.py").read_text()
        imports = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertEqual(imports, {"dataclasses", "enum", "typing"})
        for method in ["convert_image_to_base64", "get_images", "to_gradio_chatbot"]:
            self.assertFalse(hasattr(conversation.Conversation, method))


if __name__ == "__main__":
    unittest.main()
