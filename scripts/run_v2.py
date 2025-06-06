# Copyright 2025 AllenAI. All rights reserved.
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

# Runs reward model evaluation on a best-of-n dataset

import argparse
import logging
import os
import sys

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_bon_dataset_v2,
    process_single_model,
    reroll_and_score_dataset,
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
    parser.add_argument("--revision", type=str, default=None, help="revision of model to load")
    parser.add_argument(
        "--dataset", type=str, default="allenai/reward-bench-2", help="dataset, local or from huggingface"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument("--debug", action="store_true", help="Debug on small set of examples")
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--quantized", action="store_true", help="enable quantization for models that are not quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use (default: None)",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # ------- Registering Olmo config for Olmo 2 reward models -------
    if "olmo" in args.model.lower():
        from scripts.olmo_adapter import (
            Olmo2Config,
            Olmo2ForSequenceClassification,
            OlmoeConfig,
            OlmoeForSequenceClassification,
        )

        AutoModelForSequenceClassification.register(Olmo2Config, Olmo2ForSequenceClassification)
        AutoModelForSequenceClassification.register(OlmoeConfig, OlmoeForSequenceClassification)

    # ----------------------------------------------------------------

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
        config = REWARD_MODEL_CONFIG["default_v2"]
    logger.info(f"Using reward model config: {config}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": False,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    quantized = config["quantized"] or args.quantized
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
    ):
        quantized = False
        logger.info("Disabling quantization for llama3")

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]  # todo will be needed to add PairRM and SteamSHP
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)

    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    if args.revision:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, revision=args.revision, trust_remote_code=args.trust_remote_code
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    dataset, subsets, total_completions, num_correct = load_bon_dataset_v2(
        dataset=args.dataset,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
    )

    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples, corresponding to 40 rows in unrolled dataset
    if args.debug:
        dataset = dataset.select(range(40))
        subsets = subsets[:40]
        ids = ids[:40]

        # total_completions and num_correct are not unrolled, so take first 10
        total_completions = total_completions[:10]
        num_correct = num_correct[:10]

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }

    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch_dtype,
        }

    # if attn_implementation is not specified, this falls back to Hugging Face's default
    # strategy (which chooses between sdpa and eager depending on pytorch version)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.revision:
        model = model_builder(args.model, revision=args.revision, **model_kwargs, trust_remote_code=trust_remote_code)
    else:
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
    # Run inference on custom pipelines
    ############################
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
        collate_fn=custom_collate_fn,
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

            # extract score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards[0], dict):
                scores_batch = [result["score"] for result in rewards]
            # for classes that directly output scores (custom code)
            else:
                scores_batch = rewards.float().cpu().numpy().tolist()

            scores.extend(scores_batch)

    ############################
    # Print & process results
    ############################
    # add subsets and ids back (removed so it's not handled by cuda)
    out_dataset = dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores", scores)

    # reroll dataset back to one row per instance, compressing 'text' and 'score' fields into list
    # and compute results
    out_dataset = reroll_and_score_dataset(out_dataset, total_completions, cols_to_combine=["text", "scores"])
    out_dataset = out_dataset.add_column("num_correct", num_correct)

    # get core dataset
    results_grouped = {}
    model_name = f"{args.model}-{args.revision}" if args.revision else args.model
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    chat_template = args.chat_template if not check_tokenizer_chat_template(tokenizer) else "tokenizer"
    results_grouped["chat_template"] = chat_template

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        # recompute "results" column for ties subset with different scoring method
        if subset.lower() == "ties":
            ties_subset_with_results, overall_score = process_single_model(subset_dataset)
            subset_dataset = ties_subset_with_results

            # Update the results for the ties subset in the original dataset
            ties_indices = [i for i, s in enumerate(out_dataset["subset"]) if s == "ties"]
            out_dataset_df = out_dataset.to_pandas()
            for i, ties_idx in enumerate(ties_indices):
                out_dataset_df.at[ties_idx, "results"] = ties_subset_with_results["results"][i]
            out_dataset = Dataset.from_pandas(out_dataset_df)

            print(f"{subset}: Overall score {overall_score}")
            results_grouped[subset] = overall_score
        else:
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

    ############################
    # Upload results to hub
    ############################
    sub_path = "eval-set/"
    if not args.do_not_save:
        results_url = save_to_hub(
            results_grouped,
            model_name,
            sub_path,
            args.debug,
            local_only=args.do_not_save,
            save_metrics_for_beaker=not args.disable_beaker_save,
            best_of_n=True,
        )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    # upload chosen-rejected with scores
    if not model_type == "Custom Classifier":  # custom classifiers do not return scores
        # create new json with scores and upload
        scores_dict = out_dataset.to_dict()
        scores_dict["model"] = model_name
        scores_dict["model_type"] = model_type
        scores_dict["chat_template"] = chat_template

        sub_path_scores = "eval-set-scores/"

        scores_url = save_to_hub(
            scores_dict,
            model_name,
            sub_path_scores,
            args.debug,
            local_only=args.do_not_save,
            best_of_n=True,
        )
        logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")
    else:
        logger.info("Not uploading chosen-rejected text with scores due to model compatibility")


if __name__ == "__main__":
    main()
