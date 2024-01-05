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

import argparse
import json
import logging
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, concatenate_datasets
from fastchat.conversation import get_conv_template
from huggingface_hub import upload_file
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from herm import prepare_dialogue

# data repo to upload results
EVAL_REPO = "ai2-rlhf-collab/rm-benchmark-results"
PREFS_REPO = "ai2-rlhf-collab/rm-testset-results"

EVAL_SUBSETS = [
    "alpacaeval-easy",
    "alpacaeval-hard",
    "alpacaeval-length",
    "llmbar-adver-GPTInst",
    "llmbar-adver-GPTOut",
    "llmbar-adver-manual",
    "llmbar-adver-neighbor",
    "llmbar-natural",
    "mt-bench-easy",
    "mt-bench-hard",
    "mt-bench-med",
    "refusals-dangerous",
    "refusals-offensive",
]


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="natolambert/gpt2-dummy-rm", help="path to model")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="path to non-matching tokenizer, requires --direct_load"
    )
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--direct_load", action="store_true", help="directly load model instead of pipeline")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # some models need custom code to be run
    if "oasst" in args.model or "oasst" in args.chat_template:
        from herm.models import openassistant  # noqa

        model_builder = AutoModelForSequenceClassification.from_pretrained
        pipeline_builder = pipeline
        quantized = True
    elif "Starling" in args.model or "Starling" in args.chat_template:
        from herm.models.starling import RMPipeline, build_starling_rm

        model_builder = build_starling_rm
        pipeline_builder = RMPipeline
        quantized = False
    else:
        model_builder = AutoModelForSequenceClassification.from_pretrained
        pipeline_builder = pipeline
        quantized = True

    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    ############################
    # Load dataset from ai2-rlhf-collab/rm-benchmark-dev, "filtered" split
    ############################
    logger.info("*** Load dataset ***")
    # existing dataset test sets
    if args.pref_sets:
        raw_dataset = load_dataset("allenai/pref-test-sets")
        # for each split in the dataset dict, add the column "subet" with the split name to the dataset, concat all splits

        modified_datasets = []

        # Iterate over each subset in the DatasetDict
        for subset_name, dataset in raw_dataset.items():
            # Add a new column 'subset' to the dataset with the subset name
            dataset = dataset.add_column("subset", [subset_name] * len(dataset))

            # Append the modified dataset to the list
            modified_datasets.append(dataset)

        # Concatenate all the modified datasets into one dataset
        raw_dataset = concatenate_datasets(modified_datasets)

    # our custom eval set
    else:
        raw_dataset = load_dataset("ai2-rlhf-collab/rm-benchmark-dev", split="filtered")

    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # if tokenizer.chat_template exists, use that
    if False:  # tokenizer.chat_template:
        raise NotImplementedError("TODO implement this")
        # docs https://huggingface.co/docs/transformers/main/en/chat_templating
        # dataset = raw_dataset.map(
        #     lambda x: x)
        # e.g. PairRM

    # else use FastChat to get chat template
    else:
        dataset = raw_dataset.map(
            prepare_dialogue,
            fn_kwargs={"dialogue_template": conv},
            remove_columns=["prompt", "chosen", "rejected"],
            num_proc=4
        )

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": 2048,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {"device_map": {"": current_device}}
    if args.direct_load:
        model = model_builder(args.model, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        reward_pipe = pipeline_builder(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
        )
    else:
        reward_pipe = pipeline(
            "text-classification",
            model=args.model,
            revision="main",
            model_kwargs=model_kwargs,
        )

    # def collator(data):
    #     return dict((key, [d[key] for d in data]) for key in data[0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        # collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )

    dataloader, reward_pipe = accelerator.prepare(dataloader, reward_pipe)

    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id

    results = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
        rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            score_chosen = [result["score"] for result in rewards_chosen]
            score_rejected = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            score_chosen = rewards_chosen.cpu().numpy().tolist()
            score_rejected = rewards_rejected.cpu().numpy().tolist()

        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(score_chosen, score_rejected)
        ]

    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    results = {}
    results["model"] = args.model
    results["chat_template"] = args.chat_template
    # print per subset and log into results file
    for subset in EVAL_SUBSETS:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results[subset] = num_correct / num_total

    ############################
    # Upload results to hub
    ############################
    # Save results locally (results/results.json)\
    dumped = json.dumps(results, indent=4, sort_keys=True, default=str)
    logger.info(f"Stored local JSON data {dumped}.")
    path = f"results/{args.model}.json"
    dirname = os.path.dirname(path)

    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    # remove old data
    if os.path.isfile(path):
        os.remove(path)

    with open(path, "w") as f:
        f.write(dumped)

    # Upload results as json
    if not args.do_not_save:
        scores_url = upload_file(
            path_or_fileobj=path,
            path_in_repo=f"data/{args.model}.json",
            repo_id=EVAL_REPO if not args.pref_sets else PREFS_REPO, # push to correct results repo
            repo_type="dataset",
            commit_message=f"Add reward model scores for  model {args.model}",
        )
        logger.info(f"Uploaded reward model scores to {scores_url}")


if __name__ == "__main__":
    main()
