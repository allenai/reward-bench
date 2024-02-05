"""Module for visualizing datasets and post-hoc analyses"""

from collections import Counter
from typing import List, Optional, Tuple

import datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def draw_model_source_histogram(
    dataset_name: str = "ai2-adapt-dev/rm-benchmark-dev",
    output_path: Optional[str] = None,
    keys: List[str] = ["chosen_model", "rejected_model"],
    figsize: Tuple[int, int] = (12, 8),
    normalize: bool = False,
    log_scale: bool = False,
    top_n: Optional[int] = None,
) -> "matplotlib.axes.Axes":
    """Draw a histogram of the evaluation dataset that shows completion counts between models and humans.

    dataset_name (str): the HuggingFace dataset name to source the eval dataset.
    output_path (Optional[Path]): if set, then save the figure in the specified path.
    keys (List[str]): the dataset columns to include in the histogram.
    figsize (Tuple[int, int]): control the figure size when plotting.
    normalize (bool): set to True to normalize the values based on total number completions.
    log_scale (bool): set the y-axis to logarithmic scale.
    top_n (Optional[int]): if set, then only plot the top-n models in the histogram.
    RETURNS (matplotlib.axes.Axes): an Axes class containing the histogram.
    """
    dataset = datasets.load_dataset(dataset_name, split="filtered")

    if not all(key in dataset.features for key in keys):
        raise ValueError(f"Your dataset has missing keys. Please ensure that {keys} is/are available.")

    models = []
    for example in dataset:
        for key in keys:
            model = example[key]
            models.append(model)
    counter = Counter(models)

    if normalize:
        total = sum(counter.values(), 0.0)
        for key in counter:
            counter[key] /= total

    # Draw the histogram
    fig, ax = plt.subplots(figsize=figsize)
    labels, values = zip(*counter.most_common())

    if top_n:
        labels = labels[:top_n]
        values = values[:top_n]

    indices = np.arange(len(labels))
    width = 1

    ax.bar(indices, values, width)
    ax.set_xticks(indices, labels, rotation=90)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    title = f"Source of completions ({', '.join(keys)})"

    if normalize:
        ax.set_ylim(top=1.00)
        title += " , normalized"

    if log_scale:
        ax.set_yscale("log")
        title += ", log-scale"

    if top_n:
        title += f", showing top-{top_n}"

    ax.set_title(title)
    fig.tight_layout()

    if output_path:
        print(f"Saving histogram to {output_path}")
        plt.savefig(output_path, transparent=True, dpi=120)

    return ax
