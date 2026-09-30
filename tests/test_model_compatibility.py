"""Exercise custom model adapters against the supported Transformers release offline."""

import pytest
import torch
from transformers import DebertaV2Config, LlamaConfig

from rewardbench.models.beaver import LlamaForScore
from rewardbench.models.grm import GRewardModel
from rewardbench.models.inform import INFORMForSequenceClassification
from rewardbench.models.openassistant import GPTNeoXRewardModel, GPTNeoXRewardModelConfig
from rewardbench.models.openbmb import LlamaRewardModel
from rewardbench.models.pairrm import DebertaV2PairRM
from rewardbench.models.starling import LlamaForSequenceClassification


def tiny_config(config_class=LlamaConfig, **kwargs):
    return config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        num_labels=1,
        **kwargs,
    )


@pytest.fixture
def padded_inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0], [1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
    }


@pytest.mark.parametrize(
    "model_class,output_key",
    [
        (LlamaForScore, "end_scores"),
        (GRewardModel, None),
        (INFORMForSequenceClassification, "logits"),
        (LlamaRewardModel, None),
        (LlamaForSequenceClassification, "scores"),
    ],
)
def test_llama_reward_model_forward(model_class, output_key, padded_inputs):
    model = model_class(tiny_config()).eval()
    with torch.inference_mode():
        output = model(**padded_inputs)
    scores = output[output_key] if output_key else output
    assert scores.shape[0] == 2
    assert scores.numel() == 2
    assert torch.isfinite(scores).all()


def test_gpt_neox_reward_model_forward(padded_inputs):
    model = GPTNeoXRewardModel(tiny_config(GPTNeoXRewardModelConfig)).eval()
    with torch.inference_mode():
        output = model(**padded_inputs)
    assert output.logits.shape == (2, 1)
    assert torch.isfinite(output.logits).all()


def test_pairrm_forward(padded_inputs):
    config = tiny_config(
        DebertaV2Config,
        n_tasks=1,
        drop_out=0,
        sep_token_id=5,
        source_prefix_id=1,
        cand_prefix_id=6,
        cand1_prefix_id=2,
        cand2_prefix_id=3,
    )
    model = DebertaV2PairRM(config).eval()
    with torch.inference_mode():
        output = model(**padded_inputs)
    assert output.logits.shape == (2,)
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize(
    "model_class,config_class,output_key",
    [
        (LlamaForScore, LlamaConfig, "end_scores"),
        (GRewardModel, LlamaConfig, None),
        (INFORMForSequenceClassification, LlamaConfig, "logits"),
        (LlamaRewardModel, LlamaConfig, None),
        (LlamaForSequenceClassification, LlamaConfig, "scores"),
        (GPTNeoXRewardModel, GPTNeoXRewardModelConfig, "logits"),
    ],
)
def test_custom_reward_model_checkpoint_round_trip(model_class, config_class, output_key, tmp_path, padded_inputs):
    model = model_class(tiny_config(config_class)).eval()
    model.save_pretrained(tmp_path)
    restored = model_class.from_pretrained(tmp_path).eval()
    with torch.inference_mode():
        expected = model(**padded_inputs)
        actual = restored(**padded_inputs)
    if output_key:
        expected, actual = expected[output_key], actual[output_key]
    torch.testing.assert_close(actual, expected)
