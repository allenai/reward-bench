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
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from fastchat.conversation import get_conv_template
from huggingface_hub import upload_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.utils import DPODataCollatorWithPadding

from herm import (DPOInference, prepare_dialogue,
                  prepare_dialogue_from_tokenizer)

# data repo to upload results
EVAL_REPO = "ai2-rlhf-collab/rm-benchmark-results"
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
    parser.add_argument("--ref_model", type=str, default="natolambert/gpt2-dummy-rm", help="path to model")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="path to non-matching tokenizer, requires --direct_load"
    )
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--direct_load", action="store_true", help="directly load model instead of pipeline")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

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
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    ############################
    # Load dataset from ai2-rlhf-collab/rm-benchmark-dev, "filtered" split
    ############################
    logger.info("*** Load dataset ***")
    raw_dataset = load_dataset("ai2-rlhf-collab/rm-benchmark-dev", split="filtered")

    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # if tokenizer.chat_template exists, use that
    if hasattr(tokenizer, "chat_template"):
        ex = prepare_dialogue_from_tokenizer(raw_dataset[0], tokenizer)
        # docs https://huggingface.co/docs/transformers/main/en/chat_templating
        # double up to bypass some weid bug
        dataset = raw_dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer},
        )
        dataset = dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer},
        )
    # else use FastChat to get chat template
    else:
        dataset = raw_dataset.map(
            prepare_dialogue,
            fn_kwargs={"dialogue_template": conv},
        )
        dataset = dataset.map(
            prepare_dialogue,
            fn_kwargs={"dialogue_template": conv},
        )

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    model_kwargs = {
        "load_in_8bit": True,
        "device_map": {"": current_device},
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model,
        **model_kwargs,
    )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model,
        ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
    )

    # tokenize dataset
    column_names = list(dataset.features)
    tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )

    results = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        rewards_chosen, rewards_rejected = dpo.inference_step(batch)
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
            repo_id=EVAL_REPO,
            repo_type="dataset",
            commit_message=f"Add reward model scores for  model {args.model}",
        )
        logger.info(f"Uploaded reward model scores to {scores_url}")


if __name__ == "__main__":
    main()
