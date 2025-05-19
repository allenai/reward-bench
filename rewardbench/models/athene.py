# Copyright 2024 AllenAI. All rights reserved.
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
"""Implementation for Nexusflow/Athene-RM-70B."""

import math
from typing import Dict, List

import torch
from torch import nn
from transformers import LlamaModel, LlamaPreTrainedModel


class AtheneForSequenceClassification(LlamaPreTrainedModel):
    """Minimal reward model used by ``Nexusflow/Athene-RM-70B``."""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        # CLS token added by the tokenizer's chat template
        self.CLS_ID = 128003
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        rewards = self.v_head(hidden_states).squeeze(-1)

        scores = []
        bs = int(input_ids.shape[0])
        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            c_ind = c_inds[-1].item()
            scores.append(rewards[i, c_ind])
        scores = torch.stack(scores)
        return {"scores": scores}


class AtheneRewardPipeline:
    """Pipeline wrapper mimicking the usage from the model card."""

    def __init__(self, task: str, model: AtheneForSequenceClassification, tokenizer):
        self.task = task
        self.model = model.eval()
        self.tokenizer = tokenizer

    def __call__(self, samples: List[List[Dict[str, str]]], **kwargs) -> torch.Tensor:
        batch_size = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 4096)

        formatted = [self.tokenizer.apply_chat_template(s, tokenize=False) + self.tokenizer.cls_token for s in samples]
        encodings = self.tokenizer(
            formatted,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        out = []
        with torch.no_grad():
            for i in range(math.ceil(len(samples) / batch_size)):
                outputs = self.model(
                    input_ids=input_ids[i * batch_size : (i + 1) * batch_size],
                    attention_mask=attention_mask[i * batch_size : (i + 1) * batch_size],
                )
                out.append(outputs["scores"])
        return torch.hstack(out)
