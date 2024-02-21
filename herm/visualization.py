# Copyright 2023 AllenAI. All rights reserved.
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

# Module for visualizing datasets and post-hoc analyses.

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_per_token_reward(
    tokens: List[str],
    rewards: List[List[float]],
    model_names: List[str],
    font_size: int = 12,
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 12),
    line_chart: bool = False,
) -> "matplotlib.axes.Axes":
    """Draw a heatmap that combines the rewards

    tokens (List[str]): the canonical tokens that was used as reference during alignment.
    rewards (List[List[float]]): list of rewards-per-token for each model.
    model_names (List[str]): list of models
    font_size (int)
    output_path (Optional[Path]): if set, then save the figure in the specified path.
    figsize (Tuple[int, int]): control the figure size when plotting.
    line_chart (bool): if set, will draw a line chart instead of a figure.
    RETURNS (matplotlib.axes.Axes): an Axes class containing the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    matplotlib.rcParams.update(
        {
            "font.size": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        }
    )
    rewards = np.array(rewards)
    if not line_chart:
        im = ax.imshow(
            rewards,
            cmap="viridis",
            vmax=np.max(rewards),
            vmin=np.min(rewards),
        )
        fig.colorbar(im, ax=ax, orientation="horizontal", aspect=20, location="bottom")
        ax.set_xticks(np.arange(len(tokens)), [f'"{token}"' for token in tokens])
        ax.set_yticks(np.arange(len(model_names)), model_names)

        # Add text
        avg = np.mean(rewards)
        for i in range(len(model_names)):
            for j in range(len(tokens)):
                color = "k" if rewards[i, j] >= avg else "w"
                ax.text(j, i, round(rewards[i, j], 4), ha="center", va="center", color=color)

        # Make it look better
        ax.xaxis.tick_top()
        ax.tick_params(left=False, top=False)
        ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
    else:
        print("Drawing line chart")
        idxs = np.arange(0, len(tokens))
        for model_name, per_token_rewards in zip(model_names, rewards):
            ax.plot(idxs, per_token_rewards, label=model_name, marker="x")

        ax.legend(loc="upper left")
        ax.set_xticks(np.arange(len(tokens)), [f'"{token}"' for token in tokens])
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Reward")
        ax.spines[["right", "top"]].set_visible(False)

    # Added information
    title = "Cumulative substring rewards"
    ax.set_title(title, pad=20)

    # fig.tight_layout()
    if not line_chart:
        fig.subplots_adjust(left=0.5)
    if output_path:
        print(f"Saving per-token-reward plot to {output_path}")
        plt.savefig(output_path, transparent=True, dpi=120)

    plt.show()


def print_model_statistics(
    dataset_name: str = "ai2-adapt-dev/rm-benchmark-dev",
    keys: List[str] = ["chosen_model", "rejected_model"],
    render_latex: bool = False,
):
    """Print model counts and statistics into a Markdown/LaTeX table

    dataset_name (str): the HuggingFace dataset name to source the eval dataset.
    keys (List[str]): the dataset columns to include in the histogram.
    render_latex (bool): if True, render a LaTeX string.
    RETURNS (str): a Markdown/LaTeX rendering of a table.
    """
    dataset = datasets.load_dataset(dataset_name, split="filtered")

    models = {key: [] for key in keys}
    for example in dataset:
        for key in keys:
            model = example[key]
            models[key].append(model)
    counters = [Counter(models) for key, models in models.items()]

    # create another counter which is the sum of all in counters
    total_ctr = sum(counters, Counter())
    # create a table with model, total counter,
    # and the other counters by keys (0 if not in the sub counter)
    total_df = pd.DataFrame(total_ctr.most_common(), columns=["Model", "Total"])
    chosen_ctr, rejected_ctr = counters
    chosen_df = pd.DataFrame(chosen_ctr.most_common(), columns=["Model", "chosen_model"])
    rejected_df = pd.DataFrame(rejected_ctr.most_common(), columns=["Model", "rejected_model"])
    # merge these DataFrames into a single value
    model_statistics_df = (
        total_df.merge(chosen_df, how="left")
        .merge(rejected_df, how="left")
        .fillna(0)
        .astype({key: int for key in keys})
    )

    render_string = (
        model_statistics_df.to_latex(index=False)
        if render_latex
        else model_statistics_df.to_markdown(index=False, tablefmt="github")
    )
    print(render_string)
    print(f"\nTotal number of models involved: {len(total_ctr) - 2}")
    return render_string


def draw_model_source_histogram(
    dataset_name: str = "ai2-adapt-dev/rm-benchmark-dev",
    output_path: Optional[str] = None,
    keys: List[str] = ["chosen_model", "rejected_model"],
    figsize: Tuple[int, int] = (12, 8),
    font_size: int = 15,
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

    # set font size
    matplotlib.rcParams.update({"font.size": font_size})

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

    # define colors for the bars to alternate between blue (3a9fcb) and green (66b054)
    colors = ["#3a9fcb", "#66b054"]
    ax.bar(indices, values, width, color=colors * (len(indices) // 2 + 1))
    ax.set_xticks(indices, labels, rotation=90, fontsize=font_size - 2)  # font size is font_size - 2
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
