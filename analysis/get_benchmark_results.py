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

from analysis.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from analysis.constants import SUBSET_NAME_TO_PAPER_READY
from analysis.utils import load_results

LOCAL_DIR = "hf_snapshot_evals"


def get_args():
    parser = argparse.ArgumentParser()
    # optional arguments
    parser.add_argument(
        "--hf_evals_repo",
        type=str,
        default="allenai/reward-bench-results",
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


def get_average_over_rewardbench(
    df: pd.DataFrame,
    df_prefs: pd.DataFrame,
) -> pd.DataFrame:
    """Get average over a strict subset of reward models"""
    new_df = df.copy()
    for subset, sub_subsets in SUBSET_MAPPING.items():
        subset_cols = [col for col in new_df.columns if col in sub_subsets]
        sub_data = new_df[subset_cols].values  # take the relevant column values
        sub_counts = [EXAMPLE_COUNTS[s] for s in sub_subsets]  # take the example counts
        new_df[subset] = np.average(sub_data, axis=1, weights=sub_counts)

    data_cols = list(SUBSET_MAPPING.keys())
    keep_columns = ["model"] + ["model_type"] + data_cols
    new_df = new_df[keep_columns]

    # selected average from pref_sets
    pref_columns = ["anthropic_helpful", "anthropic_hhh", "shp", "summarize"]
    pref_data = df_prefs[pref_columns].values

    # add column test sets knowing the rows are not identical, take superset
    df_prefs["Prior Sets"] = np.nanmean(pref_data, axis=1)
    # add column Test Sets empty to new_df
    new_df["Prior Sets"] = np.nan
    # per row in new_df if model is in dataframe_prefs, add the value to new_df["Prior Sets"]
    values = []
    for i, row in new_df.iterrows():
        model = row["model"]
        if model in df_prefs["model"].values:
            values.append(df_prefs[df_prefs["model"] == model]["Prior Sets"].values[0])
            # new_df.at[i, "Prior Sets"] = dataframe_prefs[dataframe_prefs["model"] == model]["Prior Sets"].values[0]
        else:
            values.append(np.nan)

    new_df["Prior Sets"] = values

    # add total average
    data_cols += ["Prior Sets"]
    new_df["average"] = np.nanmean(new_df[data_cols].values, axis=1)

    # make average third column
    keep_columns = ["model", "model_type", "average"] + data_cols
    new_df = new_df[keep_columns]
    return new_df


def main():
    args = get_args()

    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("HF_TOKEN not found!")

    print(f"Downloading repository snapshots into '{LOCAL_DIR}' directory")
    # Load the remote repository using the HF API
    hf_evals_repo = snapshot_download(
        local_dir=Path(LOCAL_DIR) / "rewardbench",
        repo_id=args.hf_evals_repo,
        use_auth_token=api_token,
        tqdm_class=None,
        etag_timeout=30,
        repo_type="dataset",
    )
    hf_evals_df = load_results(hf_evals_repo, subdir="eval-set/", ignore_columns=args.ignore_columns)
    hf_prefs_df = load_results(hf_evals_repo, subdir="pref-sets/", ignore_columns=args.ignore_columns)

    def _multiply_numbered_cols_by(n, df, ignore: List[str] = []):
        numbered_cols = df.select_dtypes("number").columns
        df[numbered_cols] *= n
        return df

    all_results = {
        "RewardBench - Overview": _multiply_numbered_cols_by(
            100, get_average_over_rewardbench(hf_evals_df, hf_prefs_df)
        ),
        "RewardBench - Detailed": _multiply_numbered_cols_by(100, hf_evals_df),
        "Pref Sets - Overview": _multiply_numbered_cols_by(100, hf_prefs_df),
    }

    for name, df in all_results.items():
        # df.insert(0, "", range(1, 1 + len(df)))
        df = df.sort_values(by="average", ascending=False).round(1)
        df = df.rename(columns=SUBSET_NAME_TO_PAPER_READY)
        if args.render_latex:
            # Prettify: we're using openmojis instead of a model_type column
            def _prettify_model_name(row):
                model_type = row["model_type"]
                orig_name = row["model"]
                openmoji_map = {
                    "Seq. Classifier": "\sequenceclf",  # noqa
                    "Custom Classifier": "\customclf",  # noqa
                    "DPO": "\dpo",  # noqa
                }
                emoji = openmoji_map[model_type] if model_type in openmoji_map else "\\random"
                latex_name = (
                    f"\href{{https://huggingface.co/{orig_name}}}"  # noqa
                    + f"{{{emoji} {orig_name}}}".replace("_", "\_")  # noqa
                    if orig_name != "random"
                    else f"{emoji} {orig_name}"
                )

                return latex_name

            reward_model_names = df.apply(lambda x: _prettify_model_name(x), axis=1).to_list()
            df.insert(0, "Reward Model", reward_model_names)
            df = df.drop(columns=["model", "model_type"]).rename(columns={"average": "Average"})
            render_string = df.to_latex(index=False, float_format="%.1f").replace("NaN", "-")
        else:
            render_string = df.to_markdown(index=False, tablefmt="github")
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
