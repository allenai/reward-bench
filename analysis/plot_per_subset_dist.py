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

# Script for getting per subset distributions

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download

from analysis.utils import load_results

LOCAL_DIR = "./hf_snapshot_evals/"


def get_args():
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--hf_evals_repo",
        type=str,
        default="ai2-adapt-dev/HERM-Results",
        help="HuggingFace repository containing the evaluation results.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the results.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("HF_TOKEN not found!")

    print(f"Downloading repository snapshots into '{LOCAL_DIR}' directory")
    # Load the remote repository using the HF API
    hf_evals_repo = snapshot_download(
        local_dir=Path(LOCAL_DIR),
        repo_id=args.hf_evals_repo,
        ignore_patterns=["pref-sets-scores/*", "eval-set-scores/*"],
        use_auth_token=api_token,
        tqdm_class=None,
        repo_type="dataset",
    )
    hf_evals_df = load_results(hf_evals_repo, subdir="eval-set/")
    hf_prefs_df = load_results(hf_evals_repo, subdir="pref-sets/")
    generate_whisker_plot(hf_evals_df, args.output_dir, name="eval-set")
    generate_whisker_plot(hf_prefs_df, args.output_dir, name="pref-set")


def generate_whisker_plot(df, output_path, name=None):
    # remove the row with random in it from the df
    df = df[~df["model"].str.contains("random")]

    # Exclude 'model' and 'average' from the subsets
    subsets = [col for col in df.columns if col not in ["model", "average", "model_type", "xstest", "anthropic"]]
    n_subsets = len(subsets)

    # Calculate the number of rows and columns for the subplot grid
    nrows = int(n_subsets**0.5)
    ncols = int(n_subsets / nrows) + (n_subsets % nrows > 0)

    # Create a single figure and multiple subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the array to iterate easily if it's 2D

    # Generate plots for each subset
    for i, subset in enumerate(subsets):
        # if subset in ["donotanswer", "hep-cpp"]:
        #     import ipdb; ipdb.set_trace()
        # Filter data for the current subset
        subset_data = df[[subset]].values
        subset_data = subset_data[~np.isnan(subset_data)]

        # set axis ylim from 0 to 1
        axs[i].set_ylim(0, 1)

        # Generate box and whisker plot in its subplot
        # axs[i].boxplot(subset_data.values, vert=True, patch_artist=True)
        axs[i].violinplot(subset_data, vert=True, showmedians=True)  # , showextrema=True
        axs[i].set_title(subset)

        # turn off x-axis labels tick marks
        axs[i].set_xticks([])

        # axs[i].set_ylabel("Scores")
        axs[i].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)  # Remove x-tick labels

    # Adjusting spines and setting ticks visibility
    for ax_idx, ax in enumerate(axs):
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Determine if the subplot is on the bottom row or the leftmost column
        is_bottom = (ax_idx // ncols) == (nrows - 1) or nrows == 1
        is_left = (ax_idx % ncols) == 0

        # Only show x-axis labels for bottom row subplots
        ax.tick_params(axis="x", labelbottom=is_bottom)

        # Only show y-axis labels for leftmost column subplots
        ax.tick_params(axis="y", labelleft=is_left)

    # Adjust layout and aesthetics
    plt.suptitle("Per subset accuracy distribution", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
    plt.grid(False)

    # Handle any empty subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axs[j])

    # Show and/or save the plot
    plt.show()
    if output_path:
        print(f"Saving figure to {output_path}")
        # if output path doesn't exist, make it
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / (name + ".pdf"), transparent=True)


if __name__ == "__main__":
    main()
