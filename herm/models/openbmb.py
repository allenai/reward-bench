# noqa
from typing import List, Optional

import torch
import torch.nn as nn
from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template
from transformers import LlamaConfig, LlamaModel, PreTrainedModel

# UltraLM / UltraRM Chat Template
# Reference1: https://huggingface.co/openbmb/UltraLM-65b
# Reference2: https://huggingface.co/openbmb/UltraRM-13b
register_conv_template(
    Conversation(
        name="openbmb",
        system_message="",
        roles=("User:", "Assistant"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
    )
)


class OpenBMBPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
        batch_size = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward(  # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards
