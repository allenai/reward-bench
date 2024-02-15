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

# Script to output the per-token reward across a piece of text given a reward model

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    pipeline,
)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs='*', default="natolambert/gpt2-dummy-rm", help="path to model")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="path to non-matching tokenizer, requires --direct_load"
    )
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for inference (if above number of tokens)"
    )
    parser.add_argument("--text", type=str, default="I love to drink coffee at work.", help="text to evaluate")
    args = parser.parse_args()

    if "PairRM" in args.model or "PairRM" in args.chat_template or "SHP" in args.model or "SHP" in args.chat_template:
        # Note: SHP could be used in single-output mode, but the code is not yet added
        raise ValueError("PairRM and SHP require pairwise inputs, not supported")
    return args

def visualize_rewards(models, tokens_list_all, rewards_list_all):
    print ("HERE")
    all_scores = np.concatenate(rewards_list_all)
    vmin = np.min(all_scores)
    vmax = np.max(all_scores)
    # Create subplots with shared y-axis
    num_tokens_lists = len(tokens_list_all)
    fig, axs = plt.subplots(nrows=num_tokens_lists, figsize=(10, 2 * num_tokens_lists), sharey=True)

    # Create a single color bar for both subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    for i in range(num_tokens_lists):
        heatmap = sns.heatmap(np.array([rewards_list_all[i]]), cmap='viridis', annot=True, fmt='g',
                            xticklabels=tokens_list_all[i], yticklabels=False, ax=axs[i], vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)

        # Set labels and title
        axs[i].set_xlabel(f'{models[i]}')
        axs[i].set_ylabel('Scores')
    fig.subplots_adjust(hspace=0.4)
    title_string = f'Visualizing sub-string level rewards (note: cumulative score until that token)'
    fig.suptitle(f'{title_string}', fontsize=10)
    plt.savefig("per-token_reward_vis.png")


def tokenify_string(string, tokenizer):
    # Tokenize the entire text
    tokens = tokenizer.tokenize(string)

    cumulative_texts = []
    non_cumulative_texts = []
    # Iterate over each token
    for i, _ in enumerate(tokens):
        # Append the current cumulative text to the list
        cumulative_texts.append(tokenizer.convert_tokens_to_string(tokens[: i + 1]))
        non_cumulative_texts.append(tokenizer.convert_tokens_to_string([tokens[i]]))

    return non_cumulative_texts

def main():
    args = get_args()
    quantized = True  # only Starling isn't quantized for now
    custom_dialogue = False
    
    models = args.model

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

    tokens_list_all, rewards_list_all = list(), list()

    for model in models:
        # some models need custom code to be run
        if "oasst" in model or "oasst" in args.chat_template:
            from herm.models import openassistant  # noqa

            model_builder = AutoModelForSequenceClassification.from_pretrained
            pipeline_builder = pipeline
        elif "Starling" in model or "Starling" in args.chat_template:
            from herm.models.starling import StarlingPipeline, build_starling_rm

            model_builder = build_starling_rm
            pipeline_builder = StarlingPipeline
            quantized = False
        elif "openbmb" in model or "openbmb" in args.chat_template:
            from herm.models.openbmb import LlamaRewardModel, OpenBMBPipeline

            model_builder = LlamaRewardModel.from_pretrained
            pipeline_builder = OpenBMBPipeline
        elif "PairRM" in model or "PairRM" in args.chat_template:
            from herm.models.pairrm import DebertaV2PairRM, PairRMPipeline

            custom_dialogue = True
            model_builder = DebertaV2PairRM.from_pretrained
            pipeline_builder = PairRMPipeline
        elif "SHP" in model or "SHP" in args.chat_template:
            from herm.models.shp import SHPPipeline

            custom_dialogue = True
            model_builder = T5ForConditionalGeneration.from_pretrained
            pipeline_builder = SHPPipeline
        else:
            model_builder = AutoModelForSequenceClassification.from_pretrained
            pipeline_builder = pipeline

        if custom_dialogue:
            raise ValueError("Custom dialogue formatting not yet supported in this script")        

        logger.info(f"Running reward model on {model} with chat template {args.chat_template}")

        ############################
        # Load reward model pipeline
        ############################
        tokenizer_path = args.tokenizer if args.tokenizer else model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
        # TODO remove direct load logic
        # if pipeline_builder is pipeline, use built in pipeline, else custom
        if not pipeline == pipeline_builder:
            model = model_builder(model, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            reward_pipe = pipeline_builder(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
            )
        else:
            reward_pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                revision="main",
                model_kwargs=model_kwargs,
            )

        ############################
        # Tokenization settings & dataset preparation
        ############################
        # set pad token to eos token if not set
        if reward_pipe.tokenizer.pad_token_id is None:
            reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
            reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id


        substrings = tokenify_string(args.text, tokenizer)
        # create dataset from list of strings substrings with huggingface
        dataset = [{"text": substring} for substring in substrings]
        dataset = Dataset.from_list(dataset)

        ############################
        # Run inference [1/2]" built in transformers
        ############################
        # if using HF pipeline, can pass entire dataset and get results
        # first, handle custom pipelines that we must batch normally
        if not pipeline_builder == pipeline:
            logger.info("*** Running forward pass via built in pipeline abstraction ***")
            # this setup can be optimized slightly with one pipeline call
            # prepare for inference
            reward_pipe = accelerator.prepare(reward_pipe)

            rewards = reward_pipe(dataset["text"], **reward_pipeline_kwargs)

        ############################
        # Run inference [2/2] custom pipelines
        ############################
        else:
            logger.info("*** Running dataloader to collect results ***")

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                collate_fn=None,
                shuffle=False,
                drop_last=False,
            )

            dataloader, model = accelerator.prepare(dataloader, reward_pipe.model)
            reward_pipe.model = model

            results = list()
            for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
                logger.info(f"RM inference step {step}/{len(dataloader)}")
                rewards = reward_pipe(batch["text"], **reward_pipeline_kwargs)

                # for each item in batch, record 1 if chosen > rejected
                # extra score from dict within batched results (e.g. logits)
                # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
                if isinstance(rewards[0], dict):
                    scores = [result["score"] for result in rewards]
                # for classes that directly output scores (custom code)
                else:
                    scores = rewards.cpu().numpy().tolist()

                results.extend(scores)

        # print the results
        tokens_per_model, rewards_per_model = list(), list()
        for i, substring in enumerate(substrings):
            reward_so_far = round(results[i], 3)
            print(f"Reward: {reward_so_far} | Substring: {substring}")
            tokens_per_model.append(substring)
            rewards_per_model.append(reward_so_far)
        tokens_list_all.append(tokens_per_model)
        rewards_list_all.append(rewards_per_model)
    visualize_rewards(models, tokens_list_all, rewards_list_all)


if __name__ == "__main__":
    main()
