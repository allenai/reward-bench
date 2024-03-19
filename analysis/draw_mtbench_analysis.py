import matplotlib.pyplot as plt
import typer
from datasets import load_dataset

from analysis.visualization import AI2_COLORS, PLOT_PARAMS

plt.rcParams.update(PLOT_PARAMS)


def main():
    mtbench_url = (
        "https://huggingface.co/spaces/lmsys/mt-bench/resolve/main/data/mt_bench/model_judgment/gpt-4_single.jsonl"
    )
    mtbench_data = load_dataset("json", data_files=mtbench_url, split="train")
    single_turn = mtbench_data.filter(lambda x: x["judge"][1] == "single-v1")
    scores = {score: single_turn.filter(lambda x: x["score"] == score).num_rows for score in range(1, 10 + 1)}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(scores.keys(), scores.values(), color=AI2_COLORS.get("light_blue"))
    ax.set_xlabel("MTBench Score")
    ax.set_ylabel("Number of examples")

    ax.set_xticks(range(1, 10 + 1))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig("mtbench_scores.pdf", transparanet=True, dpi=120)


if __name__ == "__main__":
    typer.run(main)
