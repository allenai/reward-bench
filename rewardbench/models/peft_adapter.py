# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification


def build_peft_reward_model(model_name_or_path: str, **kwargs: Any) -> torch.nn.Module:
    """Load a sequence-classification adapter without losing its modules-to-save head."""
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as error:
        raise ImportError("PEFT reward adapters require installing rewardbench[peft].") from error

    adapter_revision = kwargs.pop("revision", None)
    hub_kwargs = {
        key: kwargs[key]
        for key in ("token", "cache_dir", "local_files_only", "force_download")
        if key in kwargs
    }
    adapter_config = PeftConfig.from_pretrained(
        model_name_or_path, revision=adapter_revision, **hub_kwargs
    )
    if not adapter_config.base_model_name_or_path:
        raise ValueError("The PEFT reward adapter must identify its base model.")
    base = AutoModelForSequenceClassification.from_pretrained(
        adapter_config.base_model_name_or_path, revision=adapter_config.revision, **kwargs
    )
    return PeftModel.from_pretrained(
        base,
        model_name_or_path,
        revision=adapter_revision,
        config=adapter_config,
        is_trainable=False,
        **hub_kwargs,
    )
