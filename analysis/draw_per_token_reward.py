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

# Draw the per token reward

import json
from pathlib import Path

import argparse

from herm.visualization import draw_per_token_reward

DEFAULT_DIRNAME = "per-token-reward"


def get_args():
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("text_hash", type=str, help="Path or pointer to the text hash to plot.")
    parser.add_argument("output_path", type=Path, help="Filepath to save the generated figure.")
    # optional arguments
    parser.add_argument(
        "--local",
        action="store_true",
        help="Find the file locally.",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[12, 8],
        help="Control the figure size when plotting.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.local:
        input_dir = Path.cwd() / DEFAULT_DIRNAME / args.text_hash
        assert input_dir.exists(), f"Directory {input_dir} does not exist!"

        rewards = {}
        for file in input_dir.glob("*.json"):
            with open(file) as f:
                results = json.load(f)
                rewards[results["model"]] = results

        assert len(rewards) > 0, f"Directory {input_dir} is empty!"

    else:
        # TODO: Source from a huggingface repo
        ...

    breakpoint()


if __name__ == "__main__":
    main()
