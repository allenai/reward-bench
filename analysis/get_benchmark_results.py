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
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download

LOCAL_DIR = "hf_snapshot_evals"


def get_args():
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--hf_evals_repo",
        type=str,
        default="ai2-adapt-dev/rm-benchmark-results",
        help="HuggingFace repository containing the evaluation results.",
    )
    parser.add_argument(
        "--hf_prefs_repo",
        type=str,
        default="ai2-adapt-dev/rm-testset-results",
        help="HuggingFace repository containing the test set results.",
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


def load_results(repo_dir_path: Union[str, Path], ignore_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load results into a pandas DataFrame"""
    data_dir = Path(repo_dir_path) / "data"
    orgs_dir = {d.name: d for d in data_dir.iterdir() if d.is_dir()}
    # Get all files within the subfolder orgs
    model_result_files = {d: list(path.glob("*.json")) for d, path in orgs_dir.items()}

    _results: List[pd.DataFrame] = []  # will merge later
    for org, filepaths in model_result_files.items():
        for filepath in filepaths:
            _results.append(pd.DataFrame(load_dataset("json", data_files=str(filepath), split="train")))
    results_df = pd.concat(_results)

    # Cleanup the dataframe for presentation
    def _cleanup(df: pd.DataFrame) -> pd.DataFrame:
        # remove chat_template comlumn
        df = df.drop(columns=["chat_template"])

        # move column "model" to the front
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index("model")))
        df = df.loc[:, cols]

        # select all columns except "model"
        cols = df.columns.tolist()
        cols.remove("model")
        # round
        df[cols] = df[cols].round(2)
        avg = np.nanmean(df[cols].values, axis=1).round(2)
        # add average column
        df["average"] = avg

        # move average column to the second
        cols = list(df.columns)
        cols.insert(1, cols.pop(cols.index("average")))
        df = df.loc[:, cols]

        # remove columns
        if ignore_columns:
            # Get columns from df that exist in ignore_columns
            _ignore_columns = [col for col in ignore_columns if col in df.columns]
            if len(_ignore_columns) > 0:
                print(f"Dropping columns: {', '.join(_ignore_columns)}")
                df = df.drop(_ignore_columns, axis=1)

        return df

    results_df = _cleanup(results_df)
    return results_df


def get_average_over_herm(
    df: pd.DataFrame,
    subsets: List[str] = ["alpacaeval", "mt-bench", "llmbar", "refusals", "hep"],
) -> pd.DataFrame:
    """Get average over a strict subset of reward models"""
    new_df = df.copy()
    for subset in subsets:
        if subset == "refusals":
            subset_cols = ["refusals-dangerous", "refusals-offensive", "donotanswer","xstest-should-refuse", "xstest-should-respond"]
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

    api_token = os.getenv("HF_COLLAB_TOKEN")
    if not api_token:
        raise ValueError("HF_COLLAB_TOKEN not found!")

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
    hf_evals_df = load_results(hf_evals_repo, args.ignore_columns)
    hf_prefs_repo = snapshot_download(
        local_dir=Path(LOCAL_DIR) / "prefs",
        repo_id=args.hf_prefs_repo,
        use_auth_token=api_token,
        tqdm_class=None,
        etag_timeout=30,
        repo_type="dataset",
    )
    hf_prefs_df = load_results(hf_prefs_repo, args.ignore_columns)

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
