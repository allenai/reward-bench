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

LOCAL_DIR = "hf_snapshot_evals"

import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os 
from huggingface_hub import snapshot_download
from analysis.utils import load_results

def get_args():
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--hf_evals_repo",
        type=str,
        default="ai2-adapt-dev/HERM-results",
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
        use_auth_token=api_token,
        tqdm_class=None,
        repo_type="dataset",
    )
    hf_evals_df = load_results(hf_evals_repo, args.ignore_columns)

def generate_whisker_plot(df, output_dir):
    subsets = df["subsets"].unique()

    # Generate plots for each subset
    for subset in subsets:
        # Identify columns belonging to the current subset
        subset_cols = [col for col in df.columns if subset in col]
        if not subset_cols:
            print(f"No columns found for subset '{subset}'. Skipping...")
            continue

        # Filter data for the current subset
        subset_data = df[subset_cols]

        # Generate box and whisker plot
        plt.figure(figsize=(10, 6))
        subset_data.boxplot()
        plt.title(f"Box and Whisker Plot for {subset}")
        plt.xticks(rotation=45)
        plt.ylabel("Scores")
        plt.grid(False)
        plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
        plt.show()


if __name__ == "__main__":
    main()