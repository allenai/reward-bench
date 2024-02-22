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

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset


def load_results(
    repo_dir_path: Union[str, Path], subdir: str, ignore_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load results into a pandas DataFrame"""
    base_dir = Path(repo_dir_path)
    data_dir = base_dir / subdir
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
        # if model_type is a column (pref tests may not have it)
        if "model_type" in cols:
            cols.remove("model_type")
        # remove model_beaker from dataframe
        if "model_beaker" in cols:
            cols.remove("model_beaker")
            df = df.drop(columns=["model_beaker"])

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
