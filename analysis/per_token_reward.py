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
    parser.add_argument(
        "--model",
        type=str,
        default="natolambert/gpt2-dummy-rm",
        help="path to model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="path to non-matching tokenizer, requires --direct_load",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="tulu",
        help="path to chat template",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for inference (if above number of tokens)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="I love to drink coffee at work.",
        help="text to evaluate",
    )
    args = parser.parse_args()

    if "PairRM" in args.model or "PairRM" in args.chat_template or "SHP" in args.model or "SHP" in args.chat_template:
        # Note: SHP could be used in single-output mode, but the code is not yet added
        raise ValueError("PairRM and SHP require pairwise inputs, not supported")
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
    else:
        model_builder = AutoModelForSequenceClassification.from_pretrained
        pipeline_builder = pipeline

    if custom_dialogue:
        raise ValueError("Custom dialogue formatting not yet supported in this script")

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

    ############################
    # Load reward model pipeline
    ############################
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
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

    def tokenify_string(string, tokenizer):
        # Tokenize the entire text
        tokens = tokenizer.tokenize(string)

        cumulative_texts = []
        # Iterate over each token
        for i, _ in enumerate(tokens):
            # Append the current cumulative text to the list
            cumulative_texts.append(tokenizer.convert_tokens_to_string(tokens[: i + 1]))

        return cumulative_texts

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

        results = []
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
    for i, substring in enumerate(substrings):
        print(f"Reward: {round(results[i], 3)} | Substring: {substring}")


if __name__ == "__main__":
    main()
