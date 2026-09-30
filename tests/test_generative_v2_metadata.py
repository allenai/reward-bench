"""Exercise both saved payloads without model weights, API calls, or downloads."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset


@pytest.mark.parametrize("score_w_ratings", [False, True])
def test_saved_results_record_scoring_protocol(monkeypatch, score_w_ratings):
    answer = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    dataset = Dataset.from_dict(
        {
            "texts_chosen": [[answer], [answer]],
            "texts_rejected": [[answer, answer, answer], [answer, answer, answer]],
            "id": ["0", "1"],
            "subset": ["Factuality", "Ties"],
            "num_correct": [1, 1],
        }
    )
    saved = {}

    def save_to_hub(payload, model, target_path, *args, **kwargs):
        # Round-trip the actual runner payload through its JSON representation.
        saved[target_path] = json.loads(json.dumps(payload))

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Mock(), SamplingParams=Mock()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Mock()))
    monkeypatch.setitem(sys.modules, "fastchat", None)
    monkeypatch.setitem(sys.modules, "fastchat.conversation", None)
    monkeypatch.setitem(
        sys.modules,
        "rewardbench",
        SimpleNamespace(
            load_eval_dataset_multi=lambda **kwargs: dataset,
            process_single_model=lambda data: (data.add_column("results", [None]), 0.5),
            save_to_hub=save_to_hub,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rewardbench.random_utils",
        SimpleNamespace(generate_shuffle_positions=lambda count, seed: [0] * count),
    )
    four_way = Mock(return_value=("A", None, None))
    ratings = Mock(return_value=([0], None, {"ratings": [4, 1, 1, 1]}))
    monkeypatch.setitem(
        sys.modules,
        "rewardbench.generative_v2",
        SimpleNamespace(
            ANTHROPIC_MODEL_LIST=[],
            API_MODEL_LIST=["test-model"],
            GEMINI_MODEL_LIST=[],
            OPENAI_MODEL_LIST=[],
            format_judge_answers=Mock(),
            get_single_rating=Mock(),
            process_judgement=Mock(),
            run_judge_four=four_way,
            run_judge_ratings_multi=ratings,
        ),
    )
    path = Path(__file__).parents[1] / "scripts" / "run_generative_v2.py"
    spec = importlib.util.spec_from_file_location("metadata_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [str(path), "--model", "test-model", "--do_not_save", "--disable_beaker_save"]
    if score_w_ratings:
        argv.append("--score_w_ratings")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    assert set(saved) == {"eval-set/", "eval-set-scores/"}
    for payload in saved.values():
        assert payload["score_w_ratings"] is score_w_ratings
        assert payload["model"] == "test-model"
    assert saved["eval-set/"]["Factuality"] == 1.0
    assert saved["eval-set/"]["Ties"] == 0.5
    assert saved["eval-set-scores/"]["id"] == ["0", "1"]
    assert four_way.call_count == (0 if score_w_ratings else 1)
    assert ratings.call_count == (2 if score_w_ratings else 1)
