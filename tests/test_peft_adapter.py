# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rewardbench.models import REWARD_MODEL_CONFIG
from rewardbench.models.peft_adapter import build_peft_reward_model


def test_peft_loader_keeps_base_and_adapter_revisions_separate(monkeypatch):
    base = torch.nn.Linear(1, 1)
    wrapped = torch.nn.Linear(1, 1)
    config = SimpleNamespace(base_model_name_or_path="owner/base", revision="base-sha")
    config_loader = Mock(return_value=config)
    adapter_loader = Mock(return_value=wrapped)
    base_loader = Mock(return_value=base)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftConfig=SimpleNamespace(from_pretrained=config_loader),
            PeftModel=SimpleNamespace(from_pretrained=adapter_loader),
        ),
    )
    monkeypatch.setattr(
        "rewardbench.models.peft_adapter.AutoModelForSequenceClassification.from_pretrained",
        base_loader,
    )
    result = build_peft_reward_model(
        "owner/adapter",
        revision="adapter-sha",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
        local_files_only=True,
    )
    assert result is wrapped
    config_loader.assert_called_once_with(
        "owner/adapter", revision="adapter-sha", local_files_only=True
    )
    base_loader.assert_called_once_with(
        "owner/base",
        revision="base-sha",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
        local_files_only=True,
    )
    adapter_loader.assert_called_once_with(
        base,
        "owner/adapter",
        revision="adapter-sha",
        config=config,
        is_trainable=False,
        local_files_only=True,
    )


def test_peft_loader_rejects_missing_base(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                    base_model_name_or_path=None, revision=None
                )
            ),
            PeftModel=SimpleNamespace(),
        ),
    )
    with pytest.raises(ValueError, match="identify its base"):
        build_peft_reward_model("owner/invalid-adapter")


@pytest.mark.parametrize(
    "model_id",
    [
        "Hualin3/ate-reward-qwen3-8b-grpo-20261020",
        "Hualin3/ate-reward-qwen3-8b-supervised-listwise-v4-20261022",
    ],
)
def test_ate_registration_does_not_change_default_loader(model_id):
    model = REWARD_MODEL_CONFIG[model_id]
    assert model["model_builder"] is build_peft_reward_model
    assert model["quantized"] is False
    assert model["model_type"] == "Seq. Classifier"
    assert model["torch_dtype"] is torch.bfloat16
    assert REWARD_MODEL_CONFIG["default_v2"]["model_builder"] is not build_peft_reward_model
