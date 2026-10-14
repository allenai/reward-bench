import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


def _load_generative_v2():
    module_path = Path(__file__).parents[1] / "rewardbench" / "generative_v2.py"
    spec = importlib.util.spec_from_file_location("rewardbench_generative_v2_anthropic", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_anthropic(content, calls):
    """An `anthropic` module like the 1.x SDK: no HUMAN_PROMPT / AI_PROMPT constants."""
    module = types.ModuleType("anthropic")

    class APIError(Exception):
        pass

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(content=content)

    class Anthropic:
        def __init__(self, api_key=None):
            self.messages = Messages()

    module.APIError = APIError
    module.Anthropic = Anthropic
    return module


def _block(type_, **fields):
    return types.SimpleNamespace(type=type_, **fields)


def test_anthropic_judge_works_without_text_completions_constants():
    calls = []
    fake = _fake_anthropic([_block("text", text=" [[A]] ")], calls)
    generative_v2 = _load_generative_v2()

    with patch.dict(sys.modules, {"anthropic": fake}):
        output = generative_v2.chat_completion_anthropic(
            "claude-sonnet-4-20250514",
            [{"role": "system", "content": "judge"}, {"role": "user", "content": "compare"}],
            temperature=0,
            max_tokens=16,
            api_dict={"api_key": "x"},
        )

    assert output == "[[A]]"
    assert "stop_sequences" not in calls[0]
    assert "temperature" not in calls[0]
    assert calls[0]["extra_body"] == {"temperature": 0}
    assert calls[0]["system"] == "judge"


def test_anthropic_judge_reads_the_text_block_after_a_thinking_block():
    fake = _fake_anthropic([_block("thinking", thinking="...", signature="s"), _block("text", text="[[B]]")], [])
    generative_v2 = _load_generative_v2()

    with patch.dict(sys.modules, {"anthropic": fake}):
        output = generative_v2.chat_completion_anthropic(
            "claude-sonnet-4-20250514",
            [{"role": "user", "content": "compare"}],
            temperature=0,
            max_tokens=16,
            api_dict={"api_key": "x"},
        )

    assert output == "[[B]]"


def test_chat_completion_reads_the_text_block_after_a_thinking_block():
    calls = []
    fake = _fake_anthropic(
        [_block("thinking", thinking="...", signature="s"), _block("text", text="rated\n\n9")], calls
    )
    generative_v2 = _load_generative_v2()

    with patch.dict(sys.modules, {"anthropic": fake}):
        output = generative_v2.chat_completion(
            "claude-sonnet-4-20250514",
            [{"role": "user", "content": "rate"}],
            max_tokens=16,
        )

    assert output == "rated\n\n9"
    assert "temperature" not in calls[0]
    assert calls[0]["extra_body"] == {"temperature": 0.0}
