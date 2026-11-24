# Copyright 2026 AllenAI / Minimodels contributors. Apache-2.0 License.
"""RewardBench 2 sequence-classifier pipeline & selective generative CoT runoff judge for Decision-1."""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .pipeline import disable_dropout_in_model

REFUSAL_RE = re.compile(
    r"(?:\bI\s+(?:cannot|can't|can\s+not|won't|will\s+not|am\s+unable\s+to|must\s+decline|"
    r"do\s+not\s+feel\s+comfortable)\b"
    r"|\bI'm\s+(?:sorry|unable|afraid\s+I\s+can't|not\s+able)\b"
    r"|\bI\s+apologize,\s+but\s+I\b"
    r"|\bAs\s+an\s+AI(?:\s+assistant|\s+language\s+model)?,\s+I\s+(?:cannot|can't|do\s+not|must)\b)",
    re.IGNORECASE,
)


def has_refusal_contrast(resp_a: str, resp_b: str) -> bool:
    """Return True when either candidate opens with a refusal phrase (`Opt 4v2` calibrated safety gate)."""
    return bool(REFUSAL_RE.search((resp_a or "")[:450]) or REFUSAL_RE.search((resp_b or "")[:450]))


def _extract_prompt_and_completions(batch: dict[str, Any]) -> Tuple[str, List[str], int]:
    chosen_convs = list(batch["texts_chosen"])
    rejected_convs = list(batch["texts_rejected"])
    num_correct = int(batch.get("num_correct", len(chosen_convs)))
    first_conv = chosen_convs[0]
    prompt = str(first_conv[0]["content"])
    all_convs = chosen_convs + rejected_convs
    completions = [str(conv[-1]["content"]) for conv in all_convs]
    return prompt, completions, num_correct


class DecisionRMPipeline:
    """Custom `run_v2.py` pipeline for `minimodels-ai/decision-1-reward` (`Seq. Classifier`).

    Preserves `custom_dialogue=False` compatibility with `scripts/run_v2.py` while lifting the
    default `2048` right-truncation cap to `8192` tokens so long completions keep their rubric tail.
    """

    def __init__(self, task: str, model: torch.nn.Module, tokenizer: Any) -> None:
        self.task = task
        self.model = disable_dropout_in_model(model).eval()
        self.tokenizer = tokenizer
        if hasattr(self.model, "set_tokenizer"):
            self.model.set_tokenizer(tokenizer)

    @property
    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, samples: List[str], return_inputs: bool = False, **kwargs: Any) -> Any:
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        req_max_len = int(kwargs.get("max_length", 2048) or 2048)
        max_length = max(req_max_len, 8192)

        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        if return_inputs:
            return outputs.logits, inputs
        return outputs.logits


class DecisionRMGenerativeJudge:
    """Selective Refusal-Gated Top-2 Swap+CoT (`Scratchpad:`) judge for `scripts/run_generative_v2.py`."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        tau: float = 0.80,
        w_pair: float = 0.65,
        dual_score_weight: float = 0.0,
        b_len_noul: Optional[float] = None,
        gate_refusals: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tau = float(tau)
        self.w_pair = float(w_pair)
        self.dual_score_weight = float(dual_score_weight)
        self.b_len_noul = b_len_noul
        self.gate_refusals = bool(gate_refusals)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        torch_dtype: Any = torch.bfloat16,
        device_map: str = "auto",
        tau: float = 0.80,
        w_pair: float = 0.65,
        dual_score_weight: float = 0.0,
        b_len_noul: Optional[float] = None,
        gate_refusals: bool = True,
    ) -> "DecisionRMGenerativeJudge":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            tau=tau,
            w_pair=w_pair,
            dual_score_weight=dual_score_weight,
            b_len_noul=b_len_noul,
            gate_refusals=gate_refusals,
        )

    def score_completions(self, prompt: str, completions: List[str]) -> List[float]:
        return list(
            self.model.score_completions(
                self.tokenizer,
                prompt,
                completions,
                dual_score_weight=self.dual_score_weight,
                b_len_noul=self.b_len_noul,
            )
        )

    def pairwise_runoff(self, prompt: str, resp_a: str, resp_b: str) -> Tuple[float, int]:
        return self.model.pairwise_swap_cot_runoff(self.tokenizer, prompt, resp_a, resp_b)

    def judge_non_ties_row(
        self,
        prompt: str,
        completions: List[str],
        num_correct: int = 1,
        pass1_scores: Optional[Sequence[float]] = None,
        runoff_fn: Optional[Callable[[str, str, str, int, int], Tuple[float, int]]] = None,
    ) -> Tuple[float, bool]:
        """Score a Best-of-K row; returns `(accuracy_score, runoff_triggered)`."""
        sc = list(pass1_scores) if pass1_scores is not None else self.score_completions(prompt, completions)
        order = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
        i1, i2 = order[0], order[1]
        winner_idx = i1
        triggered = False

        if self.tau > 0.0 and (sc[i1] - sc[i2]) < self.tau:
            if not (self.gate_refusals and has_refusal_contrast(completions[i1], completions[i2])):
                if runoff_fn is not None:
                    lo, _ = runoff_fn(prompt, completions[i1], completions[i2], i1, i2)
                else:
                    lo, _ = self.pairwise_runoff(prompt, completions[i1], completions[i2])
                comb = (1.0 - self.w_pair) * (sc[i1] - sc[i2]) + self.w_pair * lo
                winner_idx = i1 if comb >= 0.0 else i2
                triggered = True

        return (1.0 if winner_idx < num_correct else 0.0), triggered

    def evaluate_datasets(
        self,
        nonties_dataset: Any,
        ties_dataset: Any,
        logger: Any = None,
    ) -> Tuple[List[float], List[List[float]]]:
        """Run full RewardBench 2 non-Ties (`1,763` rows) + Ties (`102` rows) evaluation."""
        results: List[float] = []
        n_triggered = 0
        for idx, batch in enumerate(nonties_dataset):
            prompt, completions, num_correct = _extract_prompt_and_completions(batch)
            acc, trig = self.judge_non_ties_row(prompt, completions, num_correct=num_correct)
            results.append(acc)
            if trig:
                n_triggered += 1
            if logger is not None and ((idx + 1) % 200 == 0 or (idx + 1) == len(nonties_dataset)):
                logger.info(
                    f"[DecisionRM Generative] Non-ties {idx + 1}/{len(nonties_dataset)} "
                    f"(runoff_triggered={n_triggered}, tau={self.tau})"
                )

        results_ties: List[List[float]] = []
        for idx, batch in enumerate(ties_dataset):
            prompt, completions, _ = _extract_prompt_and_completions(batch)
            ratings = self.score_completions(prompt, completions)
            results_ties.append(ratings)
            if logger is not None and ((idx + 1) % 50 == 0 or (idx + 1) == len(ties_dataset)):
                logger.info(f"[DecisionRM Generative] Ties {idx + 1}/{len(ties_dataset)}")

        return results, results_ties
