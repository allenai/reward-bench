import torch
from trl import AutoModelForCausalLMWithValueHead
from transformers.utils import cached_file
from safetensors import safe_open


class SkyVLPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.tokenizer = tokenizer
        self.model = model
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model
        ).eval()
        vhead_file = cached_file(
            path_or_repo_id="Skywork/Skywork-VL-Reward-7B",
            filename="value_head.safetensors",
        )
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            vhead_params = {key: f.get_tensor(key) for key in f.keys()}
        self.model.load_state_dict(vhead_params, strict=False)
        self.model.requires_grad_(False)
        self.model.eval()

    def __call__(self, samples, **kwargs):
        inputs = self.tokenizer(
            samples, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        with torch.no_grad():
            values = self.model(**inputs, return_dict=True, use_cache=False)[-1]
            score = values.gather(
                dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
            )
            return score
