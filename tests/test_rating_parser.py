import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

RATING_CASES = [
    ("Clear and correct.\n\n8", 8),
    ("Clear and correct.\n\nRating: 8", 8),
    ("Excellent.\n\n10", 10),
    ("Excellent.\n\n10/10", 10),
    ("Clear and correct.\n\n8/10", 8),
    ("Wrong final answer.\n\nRating: 3/10", 3),
    ("Wrong final answer. I rate it 3 out of 10", 3),
    ("Wrong final answer. I RATE IT 3 OUT OF 10", 3),
    ("Mostly fine.\n\n7.5", -1),
    ("Off topic.\n\n2.5/10", -1),
    ("Harmful.\n\n-3/10", -1),
    ("Harmful.\n\n-3", -1),
    ("Out of range.\n\n11", -1),
    ("No rating given.", -1),
]


def _load(name, stubs=None):
    module_path = Path(__file__).parents[1] / "rewardbench" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rewardbench_{name}_rating", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs or {}):
        spec.loader.exec_module(module)
    return module


def _v1_import_stubs():
    """generative.py imports the API clients and fastchat at module level; stub them out."""

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        return module

    google = stub("google")
    genai = stub("google.generativeai")
    genai_types = stub("google.generativeai.types", HarmBlockThreshold=object, HarmCategory=object)
    google.generativeai = genai
    genai.types = genai_types
    return {
        "anthropic": stub("anthropic"),
        "google": google,
        "google.generativeai": genai,
        "google.generativeai.types": genai_types,
        "openai": stub("openai", OpenAI=object),
        "fastchat": stub("fastchat"),
        "fastchat.conversation": stub("fastchat.conversation", get_conv_template=lambda name: None),
        "together": stub("together", Together=object),
    }


@pytest.mark.parametrize("reply,expected", RATING_CASES)
def test_generative_v2_single_rating(reply, expected):
    generative_v2 = _load("generative_v2")
    generative_v2.chat_completion = lambda *args, **kwargs: reply

    rating, raw = generative_v2.get_single_rating("Q?", "an answer", generative_v2.OPENAI_MODEL_LIST[0])

    assert rating == expected
    assert raw == reply


@pytest.mark.parametrize("reply,expected", RATING_CASES)
def test_generative_v1_single_rating(reply, expected):
    generative = _load("generative", _v1_import_stubs())
    generative.chat_completion = lambda *args, **kwargs: reply

    rating, raw = generative._get_single_rating("Q?", "an answer", generative.OPENAI_MODEL_LIST[0])

    assert rating == expected
    assert raw == reply


def test_out_of_ten_ratings_keep_the_judges_preference():
    generative_v2 = _load("generative_v2")
    replies = {"chosen": "Correct.\n\n9/10", "r1": "Wrong.\n\n2/10", "r2": "Wrong.\n\n3/10", "r3": "Wrong.\n\n4/10"}

    def scripted_judge(model, messages, **kwargs):
        prompt = messages[-1]["content"]
        return next(reply for key, reply in replies.items() if f"ANSWER-{key}" in prompt)

    generative_v2.chat_completion = scripted_judge
    answers = [[{"role": "user", "content": "Q?"}, {"role": "assistant", "content": f"ANSWER-{k}"}] for k in replies]

    winners, _, info = generative_v2.run_judge_ratings_multi("Q?", answers, generative_v2.OPENAI_MODEL_LIST[0])

    assert info["ratings"] == [9, 2, 3, 4]
    assert winners == [0]
