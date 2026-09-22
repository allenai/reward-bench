"""Deterministic helpers for generative evaluation."""

import numpy as np


def generate_shuffle_positions(num_examples: int, seed: int) -> list[int]:
    """Return one deterministic answer position for each evaluation example."""
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")

    return np.random.default_rng(seed).integers(0, 4, size=num_examples).tolist()
