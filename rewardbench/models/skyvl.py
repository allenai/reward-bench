import torch
from safetensors import safe_open
from torch import nn
from transformers.utils import cached_file


class ValueHeadModel(nn.Module):
    """Causal LM plus a scalar value head on the last hidden state (replaces trl's AutoModelForCausalLMWithValueHead,
    which trl 1.x removed). Weight names match trl's so `value_head.safetensors` loads directly."""

    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        config = pretrained_model.config
        hidden_size = getattr(config, "hidden_size", None) or config.get_text_config().hidden_size
        self.v_head = nn.Module()
        self.v_head.summary = nn.Linear(hidden_size, 1)

    def forward(self, **kwargs):
        output = self.pretrained_model(**kwargs, output_hidden_states=True)
        hidden = output.hidden_states[-1].to(self.v_head.summary.weight.device, self.v_head.summary.weight.dtype)
        return self.v_head.summary(hidden).squeeze(-1)


class SkyVLPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.tokenizer = tokenizer
        self.model = ValueHeadModel(model).to(model.device)
        vhead_file = cached_file(
            path_or_repo_id="Skywork/Skywork-VL-Reward-7B",
            filename="value_head.safetensors",
        )
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            vhead_params = {key.removeprefix("v_head."): f.get_tensor(key) for key in f.keys()}
        self.model.v_head.load_state_dict(vhead_params)
        self.model.requires_grad_(False)
        self.model.eval()

    def __call__(self, samples, **kwargs):
        inputs = self.tokenizer(samples, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            values = self.model(**inputs, return_dict=True, use_cache=False)
            score = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
            return score
