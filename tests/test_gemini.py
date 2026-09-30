"""Offline regressions for the legacy Gemini judge's google-genai migration."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


class GeminiJudgeTest(unittest.TestCase):
    def setUp(self):
        genai = ModuleType("google.genai")
        genai_types = ModuleType("google.genai.types")
        for name in ("HttpOptions", "GenerateContentConfig", "SafetySetting"):
            setattr(genai_types, name, Mock(side_effect=lambda **values: SimpleNamespace(**values)))
        genai.types = genai_types
        google = ModuleType("google")
        google.genai = genai
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        genai.Client = Mock(return_value=self.client)
        self.genai = genai

        # Stub optional providers so no SDK, credentials, models, or network are needed.
        openai = ModuleType("openai")
        openai.OpenAI = Mock()
        together = ModuleType("together")
        together.Together = Mock()
        conversation = ModuleType("rewardbench.conversation")
        conversation.get_conv_template = Mock()
        modules = {
            "google": google,
            "google.genai": genai,
            "google.genai.types": genai_types,
            "anthropic": ModuleType("anthropic"),
            "openai": openai,
            "together": together,
            "rewardbench": ModuleType("rewardbench"),
            "rewardbench.conversation": conversation,
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        path = Path(__file__).resolve().parents[1] / "rewardbench/generative.py"
        spec = importlib.util.spec_from_file_location("rewardbench_gemini_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.API_MAX_RETRY = 2
        self.module.time = SimpleNamespace(sleep=Mock())

    def test_request_preserves_prompt_generation_and_safety_options(self):
        self.client.models.generate_content.return_value = SimpleNamespace(
            text="[[A]]", prompt_feedback=None, candidates=[]
        )
        result = self.module.chat_completion_gemini("gemini-test", "Judge this pair", 0.25, 1024)
        self.assertEqual(result, "[[A]]")
        self.client.__exit__.assert_called_once_with(None, None, None)
        client_options = self.genai.Client.call_args.kwargs
        self.assertEqual(client_options["api_key"], "test-key")
        self.assertEqual(client_options["http_options"].timeout, 1_000_000)
        self.client.models.generate_content.assert_called_once()
        request = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(request["model"], "gemini-test")
        self.assertEqual(request["contents"], "Judge this pair")
        config = request["config"]
        self.assertEqual(config.candidate_count, 1)
        self.assertEqual(config.max_output_tokens, 1024)
        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(
            {setting.category for setting in config.safety_settings},
            {
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            },
        )
        self.assertEqual({setting.threshold for setting in config.safety_settings}, {"BLOCK_NONE"})

    def test_blocked_or_empty_response_returns_error_string(self):
        self.client.models.generate_content.return_value = SimpleNamespace(
            text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"), candidates=[]
        )
        self.assertEqual(self.module.chat_completion_gemini("gemini-test", "A prompt", 0, 256), "error")

    def test_transport_failure_retries_and_returns_error(self):
        self.client.models.generate_content.side_effect = RuntimeError("simulated transport failure")
        self.assertEqual(self.module.chat_completion_gemini("gemini-test", "A prompt", 0, 256), "error")
        self.assertEqual(self.client.models.generate_content.call_count, 2)
        self.assertEqual(self.module.time.sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
