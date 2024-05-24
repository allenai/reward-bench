import random
from typing import List

import torch


class ArmoRMPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        random.seed(0)

    def __call__(self, candidates_A: List[str], candidates_B: List[str], **kwargs):
        """
        samples: List[str]
        """
        device = self.model.device
        out = []
        with torch.no_grad():
            for candidate_A, candidate_B in zip(candidates_A, candidates_B):
                pair_scores = []
                for candidate in [candidate_A, candidate_B]:
                    input_ids = self.tokenizer.apply_chat_template(candidate, return_tensors="pt").to(device)
                    output = self.model(input_ids)
                    score = output.score.float().item()
                    pair_scores.append(score)
                if pair_scores[0] == pair_scores[1]:
                    out.append(random.choice([True, False]))
                else:
                    out.append(pair_scores[0] > pair_scores[1])
        return torch.Tensor(out).bool()
