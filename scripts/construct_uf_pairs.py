from datasets import load_dataset
import numpy as np
import json
from tqdm import tqdm

dataset_dict = []
pairs = []
num_instruction = 0
count_no_overall_score = 0
count_equal_score = 0
count_missinng_completion = 0
count_missing_aspect = 0
count_sample_missing_aspect = 0
count_text_equal = 0
missing_data = []
avg_nan = 0
sources = set()



dataset = load_dataset("openbmb/UltraFeedback")["train"]

# Load TrutfulQA prompts to ensure we remove samples from evol_instruct
tqa_a = load_dataset("truthful_qa", "generation", split="validation")
tqa_b = load_dataset("truthful_qa", "multiple_choice", split="validation")

total_rows = dataset.num_rows

dataset = dataset.filter(lambda x: x["source"] != "truthful_qa", num_proc=4)
print(f"Remaining samples after removing the TruthfulQA source [{dataset.num_rows} / {total_rows}]")

contaminated_prompts = list(set(tqa_a["question"] + tqa_b["question"]))
dataset = dataset.filter(lambda x: x["instruction"] not in contaminated_prompts, num_proc=4)
print(f"Remaining samples after removing the contaminated prompts [{dataset.num_rows} / {total_rows}]")

num_instruction += len(dataset)
for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    if len(data["completions"]) < 4:
        count_missinng_completion += 1
        continue

    source = data["source"]
    instruction = data["instruction"].strip()
    completions = [data["completions"][i]["response"].encode("utf-8", "replace").decode("utf-8") for i in range(len(data["completions"]))]

    scores = []
    # for completion in data["completions"]:
    #     if "overall_score" not in completion.keys():
    #         count_no_overall_score += 1
    #         continue
    #     scores.append(completion["overall_score"])
    for completion in data["completions"]:
        if "fine-grained_score" not in completion.keys():
            count_no_overall_score += 1
            continue
        scores.append(completion["fine-grained_score"])
    

    if len(scores) == 0: # for this data, all completions are filtered by GPT-4
        # print(f"{data['completions']=}")
        continue
    if any([np.isnan(score) for score in scores]):
        sources.add(data["source"])
        avg_nan += len([score for score in scores if np.isnan(score)])
        missing_data.append(data)

    for i in range(len(completions)-1):
        for j in range(i + 1, len(completions)):
            try:
                assert scores[i] != scores[j]
            except AssertionError:
                pass
            except IndexError:
                print(i, j, len(completions), scores)
            if scores[i] == scores[j]:
                count_equal_score += 1
                continue
            else:
                chosen_text = completions[i] if scores[i] > scores[j] else completions[j]
                rejected_text = completions[j] if scores[i] > scores[j] else completions[i]
                chosen_score = max(scores[i], scores[j])
                rejected_score = min(scores[i], scores[j])

                assert chosen_score > rejected_score
                if rejected_score == 0:
                    rejected_score = 1.0
                    # print(data["completions"])
                    # exit()
                if chosen_text.strip() == rejected_text.strip():
                    count_text_equal += 1
                    continue
                pairs.append({
                    "chosen": "Human: " + instruction + "\n\n" + "Assistant: " + chosen_text,
                    "rejected": "Human: " + instruction + "\n\n" + "Assistant: " + rejected_text,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "margin": chosen_score - rejected_score,
                    "ratio": chosen_score / rejected_score,
                })

    if all([score == scores[0] for score in scores]): # all scores equal
        continue
    d = {"source": source, "instruction": instruction}
    d.update({f"completion_{i}": completion for i, completion in enumerate(completions)})
    d.update({f"score_{i}": score for i, score in enumerate(scores)})
    dataset_dict.append(d)


# with open(f"ultrafeedback_comparison_data_overall_score.json", "w") as f:
#     for data in pairs:
#         json.dump(data, f)
#         f.write("\n")

print(len(missing_data))

print(f"{len(dataset_dict)=}")
print(f"{len(pairs)=}")
print(f"{count_no_overall_score=}")
print(f"{count_equal_score=}")
print(f"{count_missinng_completion=}")
print(f"{count_text_equal=}")
print(f"{sources=}")