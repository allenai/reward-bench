# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise runner formatting and dispatch without importing GPU/API dependencies."""

import ast
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict


class TokensPrompt(TypedDict):
    """The vLLM token input schema, reproduced to keep these tests CPU-only."""

    prompt_token_ids: list[int]


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<bos>{messages[0]['content']}|{messages[-1]['content']}|<assistant>"

    def __call__(self, prompt, *, add_special_tokens, return_length):
        assert add_special_tokens is False
        assert return_length is True
        # Distinct IDs include a special token and zero; neither should be lost.
        ids = [1, 0, 10 + len(self.calls)]
        self.calls.append((prompt, ids))
        return {"input_ids": ids, "length": len(ids)}


class FakeChatTemplate:
    roles = ("user", "assistant")

    def set_system_message(self, message):
        self.system = message

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        return f"<bos>{self.system}|{self.messages[0][1]}|<assistant>"


class FakeModel:
    # Deliberately reject the removed prompt_token_ids keyword.
    def generate(self, prompts, sampling_params):
        self.prompts = prompts
        self.sampling_params = sampling_params
        return ["first output", "second output"]


def runner_nodes(script):
    """Load actual nested runner code while avoiding its heavyweight imports."""
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    tree = ast.parse(path.read_text())
    formatter = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "format_judgements"
    )
    dispatches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "model_modifier == 'Atla'"
        and any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "generate"
            for child in ast.walk(node)
        )
    ]
    assert len(dispatches) == 1
    return path, formatter, dispatches[0]


def make_batch(script, question):
    answers = [
        [{"role": "user", "content": question}, {"role": "assistant", "content": f"answer {index}"}]
        for index in range(4)
    ]
    if script == "run_generative.py":
        return {"text_chosen": answers[0], "text_rejected": answers[1]}
    return {"texts_chosen": [answers[0]], "texts_rejected": answers[1:]}


def check_formatted_prompts(script, custom_template, model_modifier):
    path, formatter, dispatch = runner_nodes(script)
    tokenizer = FakeTokenizer()
    model = FakeModel()
    sampling_params = object()
    namespace = {
        "args": SimpleNamespace(no_system_prompt=False),
        "np": SimpleNamespace(random=SimpleNamespace(rand=lambda: 0, randint=lambda *_: 0)),
        "format_judge_answers": lambda question, *_, **kwargs: ("judge system", question),
        "model_modifier": model_modifier,
        "tokenizer": tokenizer,
        "TokensPrompt": TokensPrompt,
        "model": model,
        "sampling_params": sampling_params,
        "logger": SimpleNamespace(info=lambda *_: None),
    }
    exec(compile(ast.Module(body=[formatter], type_ignores=[]), str(path), "exec"), namespace)
    template = FakeChatTemplate() if custom_template else None
    rows = [
        namespace["format_judgements"](make_batch(script, question), optional_chat_template=template)
        for question in ("first question", "second question")
    ]
    assert len(tokenizer.calls) == 2
    assert rows[0]["text"] != rows[1]["text"]
    assert [row["prompt_ids"] for row in rows] == [[1, 0, 10], [1, 0, 11]]
    assert [row["text"] for row in rows] == [prompt for prompt, _ in tokenizer.calls]
    namespace.update(prompts=[row["text"] for row in rows], prompt_ids=[row["prompt_ids"] for row in rows])
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(path), "exec"), namespace)

    expected = (
        [{"prompt_token_ids": [1, 0, 10]}, {"prompt_token_ids": [1, 0, 11]}]
        if model_modifier == "Atla"
        else [row["text"] for row in rows]
    )
    assert model.prompts == expected
    assert model.sampling_params is sampling_params
    assert namespace["outputs"] == ["first output", "second output"]


class GenerativeVllmTest(unittest.TestCase):
    def test_formatted_prompts_reach_modern_vllm(self):
        cases = product(["run_generative.py", "run_generative_v2.py"], [False, True], ["Atla", None])
        for script, custom_template, model_modifier in cases:
            with self.subTest(script=script, custom_template=custom_template, model_modifier=model_modifier):
                check_formatted_prompts(script, custom_template, model_modifier)
