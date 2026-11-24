# Copyright 2026 AllenAI / Minimodels contributors. Apache-2.0 License.
"""Unit tests for Decision-1 sequence classifier pipeline and selective generative judge."""

import torch

from rewardbench.models import REWARD_MODEL_CONFIG
from rewardbench.models.decision_rm import (
    DecisionRMGenerativeJudge,
    DecisionRMPipeline,
    has_refusal_contrast,
)


class _DummyOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _DummySeqModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.seen_max_len = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        self.seen_max_len = max(self.seen_max_len, int(input_ids.shape[1]))
        scores = attention_mask.float().sum(dim=-1, keepdim=True) * 0.01 * self.w
        return _DummyOutput(scores)


class _DummyTokenizer:
    def __call__(self, samples, truncation=True, max_length=2048, padding=True, return_tensors="pt"):
        # Verify DecisionRMPipeline lifts 2048 cap to >= 8192
        assert max_length >= 8192
        ids = torch.ones((len(samples), 6), dtype=torch.long)
        mask = torch.ones_like(ids)

        class _Batch(dict):
            def to(self, device):
                return _Batch({k: v.to(device) for k, v in self.items()})

        return _Batch({"input_ids": ids, "attention_mask": mask})


def test_decision_rm_registry_and_pipeline() -> None:
    cfg = REWARD_MODEL_CONFIG["minimodels-ai/decision-1-reward"]
    assert cfg["pipeline_builder"] is DecisionRMPipeline
    assert cfg["custom_dialogue"] is False
    assert cfg["quantized"] is False
    assert cfg["torch_dtype"] == torch.bfloat16

    pipe = DecisionRMPipeline("text-classification", _DummySeqModel(), _DummyTokenizer())
    out = pipe(["a", "b"], max_length=2048)
    assert out.shape == (2, 1)


def test_decision_rm_refusal_gated_runoff() -> None:
    assert has_refusal_contrast("Sure, here is the answer.", "I'm sorry, I cannot fulfill that request.") is True
    assert has_refusal_contrast("2 + 2 = 4.", "The sum of 2 and 2 is 4.") is False

    judge = DecisionRMGenerativeJudge(model=None, tokenizer=None, tau=0.80, w_pair=0.65, gate_refusals=True)
    # Close margin (0.20 < 0.80) on non-refusal pair triggers runoff and flips winner when log-odds < -0.15
    acc, trig = judge.judge_non_ties_row(
        "Compute 19 * 21.",
        ["399", "389", "400", "401"],
        num_correct=1,
        pass1_scores=[1.0, 1.2, -1.0, -2.0],
        runoff_fn=lambda _p, _a, _b, _i1, _i2: (-1.5, 96),  # Candidate i2 (index 0, "399") wins runoff
    )
    assert trig is True
    assert acc == 1.0

    # Refusal pair keeps Pass-1 winner without invoking runoff
    acc_ref, trig_ref = judge.judge_non_ties_row(
        "Benign prompt",
        ["Helpful answer.", "I cannot assist with that.", "Wrong.", "Off-topic."],
        num_correct=1,
        pass1_scores=[1.3, 1.0, -1.0, -2.0],
        runoff_fn=lambda *_args: (-10.0, 50),
    )
    assert trig_ref is False
    assert acc_ref == 1.0
