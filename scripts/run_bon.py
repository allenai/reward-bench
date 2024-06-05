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

# Runs best of n (BoN) ranking
# TODO: implement this for DPO models

import argparse
import logging
import os
import sys

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_bon_dataset,
    save_to_hub,
)

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--best_of", type=int, default=16, help="number of best of n to select from")
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
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

    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"]  # only Starling isn't quantized for now
    custom_dialogue = config["custom_dialogue"]
    _ = config["model_type"]  # todo will be needed to add PairRM and SteamSHP
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    dataset = load_bon_dataset(
        best_of=args.best_of,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
        remove_columns=["config", "prompt", "dataset_details", "model_input", "input"],
        # remove columns saves spave on GPU when running inference
    )
    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        ids = ids[:10]

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

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Run inference [1/2]" built in transformers
    ############################
    # if using HF pipeline, can pass entire dataset and get results
    # first, handle custom pipelines that we must batch normally
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built in pipeline abstraction ***")
        # this setup can be optimized slightly with one pipeline call
        # prepare for inference
        reward_pipe = accelerator.prepare(reward_pipe)

        results = reward_pipe(dataset["text"], **reward_pipeline_kwargs)

        # extract scores from results which is list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        scores = [r["score"] for r in results]

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
            if isinstance(batch[0]["text"][0], dict):
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

        model = accelerator.prepare(reward_pipe.model)
        reward_pipe.model = model

        scores = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            logger.info(f"RM inference step {step}/{len(dataloader)}")

            if "PairRM" in args.model or "SteamSHP" in args.model:
                raise NotImplementedError("PairRM and SteamSHP are not yet supported for batched inference")
            else:
                rewards = reward_pipe(batch["text"], **reward_pipeline_kwargs)

                # for each item in batch, record 1 if chosen > rejected
                # extra score from dict within batched results (e.g. logits)
                # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
                if isinstance(rewards[0], dict):
                    scores_batch = [result["score"] for result in rewards]
                # for classes that directly output scores (custom code)
                else:
                    scores_batch = rewards.cpu().numpy().tolist()

                scores.extend(scores_batch)

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("scores", scores)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("id", ids)

    # remove columns prompt, text, and config to save space
    # will get these from the source dataset when loading
    out_dataset = out_dataset.remove_columns("text")

    alpaca_eval = out_dataset.filter(lambda x: x["subset"] == "alpaca_eval")
    mt_bench = out_dataset.filter(lambda x: x["subset"] == "mt_bench")

    # remove subset column from both
    alpaca_eval = alpaca_eval.remove_columns("subset")
    mt_bench = mt_bench.remove_columns("subset")

    # remove model_input
    alpaca_eval = alpaca_eval.remove_columns("model_input")
    mt_bench = mt_bench.remove_columns("model_input")

    # split into per-model
    alpaca_eval_zephyr = alpaca_eval.filter(lambda x: x["model"] == "HuggingFaceH4/zephyr-7b-beta")
    alpaca_eval_tulu = alpaca_eval.filter(lambda x: x["model"] == "allenai/tulu-2-dpo-13b")
    mt_bench_zephyr = mt_bench.filter(lambda x: x["model"] == "HuggingFaceH4/zephyr-7b-beta")
    mt_bench_tulu = mt_bench.filter(lambda x: x["model"] == "allenai/tulu-2-dpo-13b")

    # def flatten and to dict
    def flatten_data(dataset):
        dictionary = dataset.to_dict()
        return [dict(zip(dictionary.keys(), values)) for values in zip(*dictionary.values())]

    ############################
    # Upload results to hub
    ############################
    sub_path = "best-of-n/"
    results_url = save_to_hub(
        flatten_data(alpaca_eval_zephyr),
        args.model,
        sub_path + "alpaca_eval/zephyr-7b/",
        args.debug,
        local_only=args.do_not_save,
    )
    results_url_2 = save_to_hub(
        flatten_data(alpaca_eval_tulu),
        args.model,
        sub_path + "alpaca_eval/tulu-13b/",
        args.debug,
        local_only=args.do_not_save,
    )
    results_url_3 = save_to_hub(
        flatten_data(mt_bench_zephyr),
        args.model,
        sub_path + "mt_bench/zephyr-7b/",
        args.debug,
        local_only=args.do_not_save,
    )
    results_url_4 = save_to_hub(
        flatten_data(mt_bench_tulu),
        args.model,
        sub_path + "mt_bench/tulu-13/",
        args.debug,
        local_only=args.do_not_save,
    )
    if not args.do_not_save:
        logger.info(
            f"Uploaded reward model results to {results_url}, {results_url_2}, {results_url_3}, {results_url_4}"
        )


if __name__ == "__main__":
    main()
