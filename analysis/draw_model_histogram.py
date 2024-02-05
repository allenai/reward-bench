"""Script to draw the distribution of model counts in a histogram"""

import argparse
from pathlib import Path

from herm.visualization import draw_model_source_histogram


def get_args():
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("output_path", type=Path, help="Filepath to save the generated figure.")
    # optional arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ai2-adapt-dev/rm-benchmark-dev",
        help="The HuggingFace dataset name to source the eval dataset.",
    )
    parser.add_argument(
        "--keys",
        type=lambda x: x.split(","),
        default="chosen_model,rejected_model",
        help="Comma-separated columns to include in the histogram.",
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[12, 8],
        help="Control the figure size when plotting.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the values based on the total number of completions.",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Set the y-axis to a logarithmic scale.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Only plot the top-n models in the histogram.",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    draw_model_source_histogram(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        keys=args.keys,
        figsize=args.figsize,
        normalize=args.normalize,
        log_scale=args.log_scale,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
