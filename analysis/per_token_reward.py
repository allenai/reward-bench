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
from typing import Any, Dict, List, Optional

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

from herm import models

REWARD_MODEL_CONFIG = {
    "default": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": pipeline,
        "quantized": True,
        "custom_dialogue": False,
    },
    "oasst": {
        "model_builder": AutoModelForSequenceClassification.from_pretrained,
        "pipeline_builder": pipeline,
        "quantized": True,
        "custom_dialogue": False,
    },
    "Starling": {
        "model_builder": models.starling.build_starling_rm,
        "pipeline_builder": models.starling.StarlingPipeline,
        "quantized": False,
        "custom_dialogue": False,
    },
    "openbmb": {
        "model_builder": models.openbmb.LlamaRewardModel.from_pretrained,
        "pipeline_builder": models.openbmb.OpenBMBPipeline,
        "quantized": True,
        "custom_dialogue": False,
    },
    "PairRM": {
        "model_builder": models.pairrm.DebertaV2Model.from_pretrained,
        "pipeline_builder": models.pairrm.PairRMPipeline,
        "quantized": True,
        "custom_dialogue": True,
    },
    "SHP": {
        "model_builder": T5ForConditionalGeneration.from_pretrained,
        "pipeline_builder": models.shp.SHPPipeline,
        "quantized": True,
        "custom_dialogue": True,
    },
}


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument(
        "text",
        type=str,
        help="Text to evaluate.",
    )
    # optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="natolambert/gpt2-dummy-rm",
        help="Path to the model or HuggingFace link.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to non-matching tokenizer, requires --direct_load.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="tulu",
        help="Path to the chat template.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (if above number of tokens).",
    )
    args = parser.parse_args()

    # Input validation
    def _validate_require_pairwise_inputs(models):
        for model in models:
            if args.model == model or args.chat_template == model:
                raise ValueError(f"{model} require pairwise inputs, not supported")

    _validate_require_pairwise_inputs(models=["PairRM", "SHP"])

    return args


def main():
    args = get_args()
    model_name = args.model if args.model in REWARD_MODEL_CONFIG.keys() else "default"
    config = REWARD_MODEL_CONFIG.get(model_name)

    if config["custom_dialogue"]:
        raise ValueError("Custom dialogue formatting not yet supported in this script")

    # Setup the accelerate state first before using logging since it errors out
    # if you do the other first.
    accelerator = Accelerator(cpu=True)
    current_device = accelerator.process_index

    # Setup logging
    logger = setup_logging(name=__name__)
    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    # Prepare dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    def _tokenify_string(string):
        tokens = tokenizer.tokenize(string)
        cumulative_texts = [tokenizer.convert_tokens_to_string(tokens[: i + 1]) for i, _ in enumerate(tokens)]
        return cumulative_texts

    substrings = _tokenify_string(args.text)
    dataset = Dataset.from_list([{"text": substring} for substring in substrings])

    # Load reward model pipeline
    logger.info("Loading reward model")
    reward_pipeline = load_reward_pipeline(
        args.model,
        config=config,
        tokenizer=tokenizer,
        process_index=current_device,
    )
    reward_pipeline_kwargs = {
        "batch_size": args.batch_size,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": 2048,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }

    # Perform inference and get per-token reward
    per_token_rewards = get_per_token_reward(
        dataset,
        reward_pipeline=reward_pipeline,
        reward_pipeline_kwargs=reward_pipeline_kwargs,
        accelerator=accelerator,
        is_custom_pipeline=config["pipeline_builder"] == pipeline,
        logger=logger,
        dataloader_batch_size=args.batch_size,
    )

    # Report the results
    for reward, token in zip(per_token_rewards, substrings):
        print(f"Reward: {round(reward, 3)} | Substring: {token}")


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    logger = get_logger(name or __name__)
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
    return logger


def load_reward_pipeline(
    model_name: str,
    *,
    config: Dict[str, Any],
    tokenizer: "transformers.PreTrainedTokenizer",
    process_index: int,
):
    model_kwargs = {"device_map": {"": process_index}}
    if config["quantized"]:
        model_kwargs.update(
            {
                "load_in_8bit": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
            }
        )
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    if not pipeline == pipeline_builder:
        reward_pipeline = pipeline_builder(
            "text-classification",
            model=model_builder(model_name, **model_kwargs),
            tokenizer=tokenizer,
        )
    else:
        reward_pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=tokenizer,
            revision="main",
            model_kwargs=model_kwargs,
        )
    # Tokenization settings
    if reward_pipeline.tokenizer.pad_token_id is None:
        reward_pipeline.model.config.pad_token_id = reward_pipeline.tokenizer.eos_token_id
        reward_pipeline.tokenizer.pad_token_id = reward_pipeline.tokenizer.eos_token_id

    return reward_pipeline


def get_per_token_reward(
    dataset: Dataset,
    *,
    reward_pipeline: "transformers.Pipeline",
    reward_pipeline_kwargs: Dict[str, Any],
    accelerator: "Accelerator",
    is_custom_pipeline: bool,
    logger: "logging.Logger",
    dataloader_batch_size: int,
) -> List[float]:
    if is_custom_pipeline:
        logger.info("Running dataloader to collect results")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_batch_size,
            collate_fn=None,
            shuffle=False,
            drop_last=False,
        )
        dataloader, model = accelerator.prepare(dataloader, reward_pipeline.model)
        reward_pipeline.model = model

        results = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")
            rewards = reward_pipeline(batch["text"], **reward_pipeline_kwargs)
            # Some pipeline implementations return a list of dictionaries, if that's the
            # case, we only take the value in the 'score' key. Else, we just return the list.
            scores = [r["score"] for r in rewards] if isinstance(rewards[0], dict) else rewards.cpu().numpy().tolist()
            results.extend(scores)
    else:
        logger.info("Running forward pass via built-in pipeline abstraction")
        reward_pipeline = accelerator.prepare(reward_pipeline)
        results = reward_pipeline(dataset["text"], reward_pipeline_kwargs)

    return results


if __name__ == "__main__":
    main()
