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
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from fastchat.conversation import get_conv_template
from huggingface_hub import upload_file
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from herm.models import REWARD_MODEL_CONFIG


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
        "--hf_dataset_repo",
        type=str,
        default="ai2-adapt-dev/per-token-reward",
        help="HuggingFace dataset repository to upload the results.",
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
        help="Path to the chat template. Will be loaded using fastchat",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="per-token-reward",
        help="Directory to store the hashes and token information.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (if above number of tokens).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Input validation
    def _validate_require_pairwise_inputs(models):
        for model in models:
            if args.model in model or args.chat_template in model:
                raise ValueError(f"{model} require pairwise inputs, not supported")

    _validate_require_pairwise_inputs(models=["PairRM", "SHP"])
    if args.hf_dataset_repo:
        assert os.getenv("HF_TOKEN")

    return args


def main():
    args = get_args()
    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        config = REWARD_MODEL_CONFIG["default"]

    if args.random_seed:
        print(f"Setting random seed to {args.random_seed}")
        torch.manual_seed(args.random_seed)

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
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _tokenify_string(string):
        _tokens = tokenizer.tokenize(string)
        cumulative_texts = [tokenizer.convert_tokens_to_string(_tokens[: i + 1]) for i, _ in enumerate(_tokens)]
        # Hacky approach. Ideally we can do a str.split(" ") but we want to
        # preserve the subword tokenization by the tokenizer.
        tokens = [tokenizer.convert_tokens_to_string([t]) for t in _tokens]
        return cumulative_texts, tokens

    # If chat_template exists
    if args.chat_template:
        print(f"Applying chat template: {args.chat_template}")
        conv = get_conv_template(args.chat_template)
        conv.append_message(role=conv.roles[0], message=args.text)
        text = conv.get_prompt()
    else:
        print("No chat template supplied.")
        text = args.text

    substrings, tokens = _tokenify_string(text)
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
    for reward, span in zip(per_token_rewards, substrings):
        print(f"Reward: {round(reward, 3)} | Substring: {span}")

    # Save the results
    save_results(
        output_dir=args.output_dir,
        text=text,
        model=args.model,
        chat_template=args.chat_template,
        substrings=substrings,
        tokens=tokens,
        rewards=per_token_rewards,
        hf_dataset_repo=args.hf_dataset_repo,
    )


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Create a logger"""
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
) -> transformers.Pipeline:
    """Load a reward model pipeline given a model configuration and its tokenizer.

    model_name (str): the HuggingFace link or pointer to the model.
    config (Dict[str, Any]): the model configuration.
    tokenizer (transformers.PreTrainedTokenizer): the tokenizer to use with the model.
    process_index (int): the machine to run the process.
    RETURNS (transformers.Pipeline) the reward model pipeline
    """
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
        model = model_builder(model_name, **model_kwargs)
        reward_pipeline = pipeline_builder(
            "text-classification",
            model=model,
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
        reward_pipeline.tokenizer.truncation_side = "left"

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
    """Get the reward per subtoken

    dataset (datasets.Dataset): the HuggingFace dataset to source the text from.
    reward_pipeline (transformers.Pipeline): the reward pipeline that will provide the scores.
    accelerator (Accelerator): accelerator class for training performance.
    is_custom_pipeline (bool): flag to check if we need to run a data loader to collate the results.
    logger (logging.Logger): logger class.
    dataloader_batch_size (int): control the batch size passed to the data loader.
    RETURNS (List[float]): list of computed rewards for each token.
    """
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


def save_results(
    output_dir: Path,
    text: str,
    model: str,
    chat_template: str,
    substrings: List[str],
    tokens: List[str],
    rewards: List[str],
    hf_dataset_repo: Optional[str] = None,
):
    """Save results to disk and on the HuggingFace hub

    This function will first hash the prompt, and then the model with the chat template.
    Then, it will save the model result in a JSON file on disk.

    output_dir (Path): directory to save the files.
    text (str): the text used to hash. The hashed string will be the name of the subdirectory.
    model (str): the name of the model
    chat_template (str): the name of the chat template.
    tokens (List[str]): the tokens extracted by the reward pipeline's tokenizer.
    rewards (List[str]): the rewards computed by the reward pipeline.
    hf_dataset_repo (Optional[str]): HuggingFace dataset to save the results to.
    """
    # Hash the text first using base16
    text_hash = hashlib.shake_256(text.encode()).hexdigest(5)
    text_dir = output_dir / text_hash
    text_dir.mkdir(parents=True, exist_ok=True)

    # Hash the model and chat_template combination
    MODEL_CHAT_DELIMITER = "___"
    model_chat_text = model + MODEL_CHAT_DELIMITER + chat_template
    model_chat_hash = hashlib.shake_256(model_chat_text.encode()).hexdigest(5)

    # Output file will be the model_chat_hash
    output_file = text_dir / f"{model_chat_hash}.json"
    print(f"Saving results to {text_dir}")

    reward_info = {
        "text": text,
        "text_hash": text_hash,
        "model": model,
        "chat_template": chat_template,
        "model_chat_hash": model_chat_hash,
        "substrings": substrings,
        "tokens": tokens,
        "rewards": rewards,
    }

    # Assumes the model output is a pointer to a HuggingFace repository
    with open(output_file, "w") as f:
        json.dump(reward_info, f, indent=4)

    # Upload to HuggingFace
    if hf_dataset_repo:
        api_token = os.getenv("HF_TOKEN")
        commit_message = (
            f"Upload {model} with template '{chat_template}' ({model_chat_hash}) results for text '{text_hash}'"
        )
        upload_file(
            repo_id=hf_dataset_repo,
            path_or_fileobj=output_file,
            path_in_repo=f"per-token-reward/{text_hash}/{model_chat_hash}.json",
            token=api_token,
            repo_type="dataset",
            commit_message=commit_message,
        )
        print(f"Saved to HuggingFace with commit message: {commit_message}")


if __name__ == "__main__":
    main()
