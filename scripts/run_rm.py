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

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    pipeline,
)

from herm import load_eval_dataset

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
api = HfApi(token=HF_TOKEN)

# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

# data repo to upload results
EVAL_REPO = "ai2-adapt-dev/rm-benchmark-results"
PREFS_REPO = "ai2-adapt-dev/rm-testset-results"


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="natolambert/gpt2-dummy-rm", help="path to model")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="path to non-matching tokenizer, requires --direct_load"
    )
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="use slow tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--direct_load", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    quantized = True  # only Starling isn't quantized for now
    custom_dialogue = False
    # some models need custom code to be run
    if "oasst" in args.model or "oasst" in args.chat_template:
        from herm.models import openassistant  # noqa

        model_builder = AutoModelForSequenceClassification.from_pretrained
        pipeline_builder = pipeline
    elif "Starling" in args.model or "Starling" in args.chat_template:
        from herm.models.starling import StarlingPipeline, build_starling_rm

        model_builder = build_starling_rm
        pipeline_builder = StarlingPipeline
        quantized = False
    elif "openbmb" in args.model or "openbmb" in args.chat_template:
        from herm.models.openbmb import LlamaRewardModel, OpenBMBPipeline

        model_builder = LlamaRewardModel.from_pretrained
        pipeline_builder = OpenBMBPipeline
    elif "PairRM" in args.model or "PairRM" in args.chat_template:
        from herm.models.pairrm import DebertaV2PairRM, PairRMPipeline

        custom_dialogue = True
        model_builder = DebertaV2PairRM.from_pretrained
        pipeline_builder = PairRMPipeline
    elif "SHP" in args.model or "SHP" in args.chat_template:
        from herm.models.shp import SHPPipeline

        custom_dialogue = True
        model_builder = T5ForConditionalGeneration.from_pretrained
        pipeline_builder = SHPPipeline
    elif "beaver" in args.model or "pku-align" in args.chat_template:
        from herm.models.beaver import BeaverPipeline, LlamaForScore

        model_builder = LlamaForScore.from_pretrained
        pipeline_builder = BeaverPipeline
    elif "Ziya" in args.model or "Ziya" in args.chat_template:
        from herm.models.ziya import ZiyaPipeline

        model_builder = AutoModelForSequenceClassification.from_pretrained
        pipeline_builder = ZiyaPipeline
        quantized = False  # handled by .half() in the custom pipeline, as in model card
    else:
        model_builder = AutoModelForSequenceClassification.from_pretrained
        pipeline_builder = pipeline

    trust_remote_code = args.trust_remote_code

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
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=not args.use_slow_tokenizer,
    )
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected"],
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
        "max_length": 1024,
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
    # TODO remove direct load logic
    # if pipeline_builder is pipeline, use built in pipeline, else custom
    if args.direct_load or not pipeline_builder == pipeline:
        model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if 'gpt2' in args.model:
            model.config.pad_token_id = model.config.eos_token_id
        reward_pipe = pipeline_builder(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
        )
    else:
        reward_pipe = pipeline(
            "text-classification",
            model=args.model,
            tokenizer=tokenizer,
            revision="main",
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
        )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    print('starting scores!!!!!!!!')
    if not args.direct_load or pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)

        results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
        results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)

        # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        score_chosen = [result["score"] for result in results_cho]
        print(score_chosen)
        score_rejected = [result["score"] for result in results_rej]
        print(score_rejected)

        # pairwise comparison list comprehension
        results = [1 if chosen > rejected else 0 for chosen, rejected in zip(score_chosen, score_rejected)]

    ############################
    # Run inference [2/2] custom pipelines
    ############################
    else:
        logger.info("*** Running dataloader to collect results ***")
        # TODO make more custom pipelines work with pre-tokenized data
        from torch.utils.data.dataloader import default_collate

        # for PairRM, hmm, will move all of this later
        def custom_collate_fn(batch):
            # check if ['text_chosen'] is in first batch element
            # Check if the first element of the batch is a dictionary
            if isinstance(batch[0]["text_chosen"][0], dict):
                return batch  # Return the batch as-is if it's a list of dicts
            else:
                return default_collate(batch)  # Use the default collate behavior otherwise

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
        reward_pipe.model = model

        results = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if (
                "PairRM" in args.model
                or "PairRM" in args.chat_template
                or "SHP" in args.model
                or "SHP" in args.chat_template
            ):
                text_rejected = [b["text_rejected"] for b in batch]
                text_chosen = [b["text_chosen"] for b in batch]
                results_sub = reward_pipe(text_chosen, text_rejected, **reward_pipeline_kwargs)
                [results.append(1) if result else results.append(0) for result in results_sub.cpu().numpy().tolist()]
            else:
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

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)
    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)

    results = {}
    results["model"] = args.model
    results["chat_template"] = args.chat_template
    # print per subset and log into results file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
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
    path = "results/metrics.json"
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
        scores_url = api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"data/{args.model}.json",
            repo_id=EVAL_REPO if not args.pref_sets else PREFS_REPO,  # push to correct results repo
            repo_type="dataset",
            commit_message=f"Add reward model scores for  model {args.model}",
        )
        logger.info(f"Uploaded reward model scores to {scores_url}")


if __name__ == "__main__":
    main()
