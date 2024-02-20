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

# Script for getting reward model benchmark results

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from analysis.utils import load_results

LOCAL_DIR = "hf_snapshot_evals"


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
    parser.add_argument(
        "--render_latex",
        action="store_true",
        help="If set, then it will render a LaTeX string instead of Markdown.",
    )
    parser.add_argument(
        "--ignore_columns",
        type=lambda x: x.split(",") if x is not None else None,
        default=None,
        help="Comma-separated column names to exclude from the report.",
    )
    args = parser.parse_args()
    return args


def get_average_over_herm(
    df: pd.DataFrame,
    subsets: List[str] = ["alpacaeval", "mt-bench", "llmbar", "refusals", "hep"],
) -> pd.DataFrame:
    """Get average over a strict subset of reward models"""
    new_df = df.copy()
    for subset in subsets:
        if subset == "refusals":
            subset_cols = [
                "refusals-dangerous",
                "refusals-offensive",
                "donotanswer",
                "xstest-should-refuse",
                "xstest-should-respond",
            ]
        else:
            subset_cols = [col for col in new_df.columns if subset in col]
        new_df[subset] = np.round(np.nanmean(new_df[subset_cols].values, axis=1), 2)

    keep_columns = ["model", "average"] + subsets
    new_df = new_df[keep_columns]
    # Replace 'average' column with new average
    new_df["average"] = np.round(np.nanmean(new_df[subsets].values, axis=1), 2)
    # Rename column "hep" to "hep (code)"
    new_df = new_df.rename(columns={"hep": "hep (code)"})
    return new_df


def main():
    args = get_args()

    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("HF_TOKEN not found!")

    print(f"Downloading repository snapshots into '{LOCAL_DIR}' directory")
    # Load the remote repository using the HF API
    hf_evals_repo = snapshot_download(
        local_dir=Path(LOCAL_DIR) / "herm",
        repo_id=args.hf_evals_repo,
        use_auth_token=api_token,
        tqdm_class=None,
        etag_timeout=30,
        repo_type="dataset",
    )
    hf_evals_df = load_results(hf_evals_repo, subdir="eval-set/", ignore_columns=args.ignore_columns)
    hf_prefs_df = load_results(hf_evals_repo, subdir="pref-sets/", ignore_columns=args.ignore_columns)

    all_results = {
        "HERM - Overview": get_average_over_herm(hf_evals_df),
        "HERM - Detailed": hf_evals_df,
        "Pref Sets - Overview": hf_prefs_df,
    }

    for name, df in all_results.items():
        df = df.sort_values(by="average", ascending=False).round(4)
        render_string = (
            df.round(4).astype(str).to_latex(index=False)
            if args.render_latex
            else df.to_markdown(index=False, tablefmt="github")
        )
        render_string = render_string.replace("NaN", "")
        render_string = render_string.replace("nan", "")
        print(name)
        print(render_string)

        if args.output_dir:
            print(f"Saving results to '{args.output_dir}/{name}.csv'")
            Path(args.output_dir).mkdir(exist_ok=True, parents=True)
            df.to_csv(args.output_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
