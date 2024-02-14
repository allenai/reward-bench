from datasets import concatenate_datasets, load_dataset

# ultrafeedback = load_dataset("openbmb/UltraFeedback")
# shp = load_dataset("stanfordnlp/SHP")
# anthropic_hh = load_dataset("Anthropic/hh-rlhf")
# openai_summarize = load_dataset("openai/summarize_from_feedback", "comparisons")

# format:
# {
#     'prompt': 'prompt',
#     'prompt_id': 'id',
#     'chosen': [
        # {
        #     'content': 'prompt',
        #     'role': 'user'
        # },
        # {
        #     'content': "chosen model output",
        #     'role': 'assistant'
        # }
#     ],
    # 'rejected': [
    #     {
    #         'content': 'prompt',
    #         'role': 'user'
    #     },
    #     {
    #         'content': "rejected model output",
    #         'role': 'assistant'
    #     }
    # ],
#     'messages': [
#           ###### this is unnecessary
#     ],
#     'score_chosen': 8.5, ###### this is unnecessary
#     'score_rejected': 7.5 ###### this is unnecessary
# }

def map_to_tulu_format(ex):
    # First make sure we end with assistant (this shouldn't be necessary for these datasets)
    if ex['chosen'][-1]['role'] != 'assistant' or ex['rejected'][-1]['role'] != 'assistant':
        return {
            'chosen': None,
            'rejected': None,
        }
    
    # Build chosen and rejected
    chosen = ""
    rejected = ""
    for message in ex['chosen']:
        chosen += f"<|{message['role']}|>\n{message['content']}\n"
    for message in ex['rejected']:
        rejected += f"<|{message['role']}|>\n{message['content']}\n"

    return {
        'chosen': chosen.strip(),
        'rejected': rejected.strip(),
    }

def get_uf_choices():
    from datasets import load_dataset, DatasetDict, concatenate_datasets
    import hashlib
    import random
    import time

    random.seed(42)

    # Load revision with the fixes to overall_score
    ds = load_dataset("openbmb/UltraFeedback", split="train")

    # Load TrutfulQA prompts to ensure we remove samples from evol_instruct
    tqa_a = load_dataset("truthful_qa", "generation", split="validation")
    tqa_b = load_dataset("truthful_qa", "multiple_choice", split="validation")

    total_rows = ds.num_rows

    ds = ds.filter(lambda x: x["source"] != "truthful_qa", num_proc=4)
    print(f"Remaining samples after removing the TruthfulQA source [{ds.num_rows} / {total_rows}]")

    contaminated_prompts = list(set(tqa_a["question"] + tqa_b["question"]))
    ds = ds.filter(lambda x: x["instruction"] not in contaminated_prompts, num_proc=4)
    print(f"Remaining samples after removing the contaminated prompts [{ds.num_rows} / {total_rows}]")

    def get_pairwise_completions(completions):
        start = time.time()
        scores_and_completions = []
        scores_and_completions = [(float(c["overall_score"]), c["response"], c["model"]) for c in completions]
        if len(scores_and_completions) < 2:
            return [(None, None)]
        pairs = []
        scores_and_completions.sort(key=lambda x: x[0], reverse=True)
        for i in range(len(scores_and_completions) - 1):
            for j in range(i + 1, len(scores_and_completions)):
                chosen = scores_and_completions[i]
                rejected = scores_and_completions[j]
                # if chosen[0] <= 1.0:
                    # continue
                if chosen[0] == rejected[0]:
                    continue
                pairs.append((chosen, rejected))
        return pairs

        chosen = max(scores_and_completions, key=lambda x: x[0])
        rejected = random.choice(scores_and_completions)
        while rejected == chosen:
            end = time.time()
            if end - start > 3:
                print("Timeout")
                print(chosen, rejected)
                break
            rejected = random.choice(scores_and_completions)
        return chosen, rejected


    # def format_prompt(x):
    #     prompt = x["instruction"]
    #     # chosen, rejected = get_pairwise_completions(x["completions"])
    #     pairs = get_pairwise_completions(x["completions"])
    #     to_return = {
    #             "prompt": [],
    #             "prompt_id": [],
    #             "chosen": [],
    #             "rejected": [],
    #             "messages": [],
    #             "score_chosen": [],
    #             "score_rejected": []
    #     }
    #     for chosen, rejected in pairs:
    #         chosen_messages = []
    #         rejected_messages = []
    #         chosen_messages = [
    #             {"role": "user", "content": prompt},
    #             {"role": "assistant", "content": chosen[1] if chosen is not None else "N/A"},
    #         ]
    #         rejected_messages = [
    #             {"role": "user", "content": prompt},
    #             {"role": "assistant", "content": rejected[1] if rejected is not None else "N/A"},
    #         ]
    #         if isinstance(prompt, list):
    #             print("prompt is list?")
    #             print(len(prompt))
    #             continue
    #         # to_return.append( {
    #         #     "prompt": prompt,
    #         #     "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    #         #     "chosen": chosen_messages,
    #         #     "rejected": rejected_messages,
    #         #     "messages": chosen_messages, # Use best-ranked example for SFT
    #         #     "score_chosen": chosen[0] if chosen is not None else -100.0,
    #         #     "score_rejected": rejected[0] if rejected is not None else -100.0,
    #         # })
    #         to_return["prompt"].append(prompt)
    #         to_return["prompt_id"].append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
    #         to_return["chosen"].append(chosen_messages)
    #         to_return["rejected"].append(rejected_messages)
    #         to_return["messages"].append(chosen_messages)
    #         to_return["score_chosen"].append(chosen[0] if chosen is not None else -100.0)
    #         to_return["score_rejected"].append(rejected[0] if rejected is not None else -100.0)
    #     return to_return

    def format_prompts(x):
        prompts = x["instruction"]
        # chosen, rejected = get_pairwise_completions(x["completions"])
        pairs = [get_pairwise_completions(pair) for pair in x["completions"]]
        to_return = {
                "prompt": [],
                "prompt_id": [],
                "chosen": [],
                "rejected": [],
                "messages": [],
                "score_chosen": [],
                "score_rejected": []
        }
        for prompt, pair_list in zip(prompts, pairs):
            for chosen, rejected in pair_list:
                chosen_messages = []
                rejected_messages = []
                chosen_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen[1] if chosen is not None else "N/A"},
                ]
                rejected_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected[1] if rejected is not None else "N/A"},
                ]
                if isinstance(prompt, list):
                    print("prompt is list?")
                    print(len(prompt))
                    continue
                # to_return.append( {
                #     "prompt": prompt,
                #     "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                #     "chosen": chosen_messages,
                #     "rejected": rejected_messages,
                #     "messages": chosen_messages, # Use best-ranked example for SFT
                #     "score_chosen": chosen[0] if chosen is not None else -100.0,
                #     "score_rejected": rejected[0] if rejected is not None else -100.0,
                # })
                to_return["prompt"].append(prompt)
                to_return["prompt_id"].append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
                to_return["chosen"].append(chosen_messages)
                to_return["rejected"].append(rejected_messages)
                to_return["messages"].append(chosen_messages)
                to_return["score_chosen"].append(chosen[0] if chosen is not None else -100.0)
                to_return["score_rejected"].append(rejected[0] if rejected is not None else -100.0)
        return to_return


    # ['source', 'instruction', 'models', 'completions', 'correct_answers', 'incorrect_answers']
    ds = ds.map(format_prompts, num_proc=1, remove_columns=ds.column_names, batched=True)


    # filter out margin = -100
    ds = ds.filter(lambda x: x["score_chosen"] != -100 or x["score_rejected"] != -100, num_proc=8)



    def remove_last_step_for_rl(example):
        example["messages"] = example["messages"][:-1]  # remove the assistant response
        return example


    all_ds = DatasetDict()

    split_dataset = ds.train_test_split(test_size=2000, seed=42, shuffle=True)
    test_datasets = split_dataset["test"].train_test_split(0.5, seed=42, shuffle=True)

    all_ds["train_prefs"] = split_dataset["train"]
    all_ds["train_sft"] = split_dataset["train"]
    # Keep more examples for test accuracy
    all_ds["test_prefs"] = concatenate_datasets([test_datasets["train"], test_datasets["test"]])
    all_ds["test_sft"] = test_datasets["train"]


    # remove empty last turns
    def filter_empty_messages(example):
        if example["messages"][-1]["role"] == "user":
            example["messages"] = example["messages"][:-1]
        if example["chosen"][-1]["role"] == "user":
            example["chosen"] = example["chosen"][:-1]
        if example["rejected"][-1]["role"] == "user":
            example["rejected"] = example["rejected"][:-1]
        return example


    all_ds = all_ds.map(filter_empty_messages)

    all_ds["train_gen"] = all_ds["train_sft"].map(remove_last_step_for_rl)
    all_ds["test_gen"] = all_ds["test_sft"].map(remove_last_step_for_rl)

    assistant_rows = []

    # check that gen split does not end with `assistant`, should print 0
    for idx, row in enumerate(all_ds["train_gen"]):
        if row["messages"][-1]["role"] == "assistant":
            assistant_rows.append(row)
    for row in all_ds["test_gen"]:
        if row["messages"][-1]["role"] == "assistant":
            assistant_rows.append(row)

    assert len(assistant_rows) == 0

    print(all_ds)

    print(len(all_ds))

    print(all_ds['train_prefs'][0])

    return all_ds["train_prefs"].map(map_to_tulu_format, remove_columns=all_ds["train_prefs"].column_names)

def get_shp_choices():
    shp = load_dataset("stanfordnlp/SHP", split="train")
    shp = shp.filter(lambda x: x["score_ratio"] >= 2.0, num_proc=8)

    def map_example(ex):
        if ex['labels'] == 0:
            chosen = ex['human_ref_B']
            rejected = ex['human_ref_A']
        elif ex['labels'] == 1:
            chosen = ex['human_ref_A']
            rejected = ex['human_ref_B']
        else:
            print(f"labels???? {ex['labels']}")
        return {
            'prompt': ex['history'],
            'prompt_id': ex['post_id'],
            'chosen': [
                {
                    'content': ex['history'],
                    'role': 'user',
                },
                {
                    'content': chosen,
                    'role': 'assistant',
                }
            ],
            'rejected': [
                {
                    'content': ex['history'],
                    'role': 'user',
                },
                {
                    'content': rejected,
                    'role': 'assistant',
                }
            ]
        }

    shp = shp.map(map_example, remove_columns=shp.column_names)

    return shp.map(map_to_tulu_format, remove_columns=shp.column_names)

def get_anthropic_hh():
    def reformat_examples(ex):
        return {
            'chosen': ex['chosen'].replace('\n\nHuman:', '\n<|user|>\n').replace('\n\nAssistant:', "\n<|assistant|>\n").strip(),
            'rejected': ex['rejected'].replace('\n\nHuman:', '\n<|user|>\n').replace('\n\nAssistant:', "\n<|assistant|>\n").strip(),
        }

    helpful = concatenate_datasets(
        [
            load_dataset("polinaeterna/hh-rlhf", "helpful-base", split="train"),
            load_dataset("polinaeterna/hh-rlhf", "helpful-online", split="train"),
            load_dataset("polinaeterna/hh-rlhf", "helpful-rejection-sampled", split="train"),
        ]
    )

    helpful = helpful.map(reformat_examples)
    # helpful_base = load_dataset("polinaeterna/hh-rlhf", "helpful-base", split="train")
    # helpful_online = load_dataset("polinaeterna/hh-rlhf", "helpful-online", split="train")
    # helpful_rs = load_dataset("polinaeterna/hh-rlhf", "helpful-rejection-sampled", split="train")

    return helpful

def get_openai_summarize():
    def map_to_format(ex):
        assert ex['choice'] in [0,1]
        if ex['choice'] == 0:
            chosen = ex['summaries'][0]
            rejected = ex['summaries'][1]
        else:
            chosen = ex['summaries'][1]
            rejected = ex['summaries'][0]

        return {
            'chosen': [
                {
                    'content': ex['info']['title'] + '\n\n' + ex['info']['post'],
                    'role': 'user'
                },
                {
                    'content': chosen['text'],
                    'role': 'assistant'
                }
            ],
            'rejected': [
                {
                    'content': ex['info']['title'] + '\n\n' + ex['info']['post'],
                    'role': 'user'
                },
                {
                    'content': rejected['text'],
                    'role': 'assistant'
                }
            ],
        }

    openai_summarize = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")
    openai_summarize = openai_summarize.map(map_to_format, remove_columns=openai_summarize.column_names).map(map_to_tulu_format)

    return openai_summarize


def get_all_datasets():
    ds = get_uf_choices()
    ds = get_shp_choices()
    ds = get_anthropic_hh()
    ds = get_openai_summarize()

    return concatenate_datasets(
        [
            get_uf_choices(),
            get_shp_choices(),
            get_anthropic_hh(),
            get_openai_summarize()
        ]
    )

with open('/net/nfs.cirrascale/allennlp/jacobm/herm/data/uf-repro/data.jsonl', 'w') as f_out:
    import json
    for elem in get_all_datasets():
        f_out.write(json.dumps(elem) + '\n')