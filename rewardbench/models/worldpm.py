import torch


class WorldPMPipeline:
    """Pipeline for Qwen WorldPM models."""

    def __init__(self, task, model, tokenizer):
        self.task = task
        # disable dropout and set eval
        self.model = model.eval()
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=inputs["input_ids"])

        if isinstance(outputs, (list, tuple)):
            scores = outputs[0]
        elif isinstance(outputs, dict):
            # take first value in dict
            scores = next(iter(outputs.values()))
        else:
            scores = outputs
        return scores
