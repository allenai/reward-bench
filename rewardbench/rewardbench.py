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

# Run RewardBench (evaluate any reward model on any dataet)

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import EvalResult, ModelCard, ModelCardData
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from rewardbench import (
    DPO_MODEL_CONFIG,
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_preference_dataset,
)


@dataclass
class Args:
    # core args
    dataset: str = "allenai/reward-bench"
    """The dataset to evaluate on."""
    split: Optional[str] = None
    """The split to evaluate on."""
    model: Optional[str] = None
    """The model to evaluate."""
    revision: Optional[str] = None
    """The model revision to evaluate."""
    ref_model: Optional[str] = None
    """The reference model to compare against."""
    tokenizer: Optional[str] = None
    """The tokenizer to use (defaults to model)."""
    chat_template: Optional[str] = None
    """The chat template to use (defaults to from tokenizer, from chattemplate)."""
    not_quantized: bool = False
    """Disable quantization for models that are quantized by default."""

    # wandb args
    wandb_run: Optional[str] = None
    """The wandb run to extract model and revision from."""
    upload_metadata_to_hf: bool = False
    """Upload metadata to Hugging Face Hub."""

    # inference args
    batch_size: int = 8
    """The batch size to use."""
    max_length: int = 512
    """The max length to use."""

    # system args
    load_json: bool = False
    """Load dataset as json."""
    trust_remote_code: bool = False
    """Trust remote code."""
    debug: bool = False
    """Debug mode."""
    output_dir: str = "results/"
    """The output directory to save results."""
    save_all: bool = False
    """Save all results."""
    force_truncation: bool = False
    """Force truncation (for if model errors)."""


def main():
    parser = HfArgumentParser((Args))
    actual_main(*parser.parse_args_into_dataclasses())


def actual_main(args: Args):
    if args.wandb_run is not None:
        wandb_run = wandb.Api().run(args.wandb_run)
        args.model = wandb_run.config["hf_repo_id"]
        args.revision = wandb_run.config["hf_repo_revision"]

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
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # basic checks from config
    if args.ref_model:
        is_dpo = True
        MODEL_CONFIGS = DPO_MODEL_CONFIG
        assert args.model != args.ref_model, "policy and reference model should be different"
        from trl.trainer.utils import DPODataCollatorWithPadding

        from rewardbench import DPOInference
    else:
        is_dpo = False
        MODEL_CONFIGS = REWARD_MODEL_CONFIG

    if args.chat_template:
        from fastchat.conversation import get_conv_template

        conv = get_conv_template(args.chat_template)
    else:
        conv = None

    if args.model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model]
    else:
        config = MODEL_CONFIGS["default"]
    logger.info(f"Using reward model config: {config}")

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    if not is_dpo:
        quantized = config["quantized"]  # only Starling isn't quantized for now
        # if llama-3 in name, switch quantized to False (severely degrades performance)
        if (
            ("llama-3" in args.model)
            or ("Llama3" in args.model)
            or ("Llama-3" in args.model)
            or ("LLaMA3" in args.model)
            or args.not_quantized
        ):
            quantized = False
            logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")
        custom_dialogue = config["custom_dialogue"]
        pipeline_builder = config["pipeline_builder"]
        _ = config["model_type"]
        torch_dtype = config.get("torch_dtype", None)
        if custom_dialogue:
            raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

    model_builder = config["model_builder"]

    #########################
    # load dataset
    #########################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code, revision=args.revision
    )
    if args.dataset == "allenai/reward-bench":
        logger.info("Running core eval dataset.")
        from rewardbench import load_eval_dataset
        from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
        from rewardbench.utils import calculate_scores_per_section

        # primary set compiles slightly more information
        dataset, subsets = load_eval_dataset(
            core_set=True,
            conv=conv,
            custom_dialogue_formatting=False,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "prompt"],
        )
    else:
        dataset = load_preference_dataset(
            args.dataset, split=args.split, json=args.load_json, tokenizer=tokenizer, conv=conv
        )

    if args.debug:
        dataset = dataset.select(range(10))

    logger.info("*** Load reward model ***")

    ############################
    # Load DPO model pipeline
    ############################
    if is_dpo:
        tokenizer.pad_token = tokenizer.eos_token
        # if no BOS token, set as pad token, e.g. QWEN models
        if tokenizer.bos_token is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        model = model_builder(
            args.model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )

        # use internal inference functions in DPO trainer
        dpo = DPOInference(
            model,
            ref_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            # norm is norm, avg is average, sum is sum
        )

        # tokenize dataset
        column_names = list(dataset.features)

        tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=args.batch_size,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )

    ############################
    # Load classifier model pipeline
    ############################
    else:

        # padding experiments for determinism
        tokenizer.padding_side = "left"
        truncation = False
        if args.force_truncation:
            truncation = True
            tokenizer.truncation_side = "left"

        reward_pipeline_kwargs = {
            "batch_size": args.batch_size,  # eval_args.inference_batch_size,
            "truncation": truncation,
            "padding": True,
            "max_length": args.max_length,
            "function_to_apply": "none",  # Compute raw logits
            "return_token_type_ids": False,
        }
        if quantized:
            if torch_dtype is not None:
                torch_dtype = torch_dtype
            else:
                torch_dtype = torch.float16
            model_kwargs = {
                "load_in_8bit": True,
                "device_map": {"": current_device},
                "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
            }
        else:
            # note, device map auto does not work for bitsandbytes quantized models
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch_dtype,
            }

        model = model_builder(
            args.model, **model_kwargs, revision=args.revision, trust_remote_code=args.trust_remote_code
        )
        reward_pipe = pipeline_builder(
            "text-classification",  # often not used
            model=model,
            tokenizer=tokenizer,
        )

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

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = accelerator.prepare(reward_pipe.model)
        reward_pipe.model = model

    ############################
    # Run inference
    ############################

    results = []
    scores_chosen = []
    scores_rejected = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        if is_dpo:
            rewards_chosen, rewards_rejected = dpo.inference_step(batch)
        else:
            rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
            rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            score_chosen_batch = [result["score"] for result in rewards_chosen]
            score_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
            score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

        # log results
        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
        ]
        scores_chosen.extend(score_chosen_batch)
        scores_rejected.extend(score_rejected_batch)

    ############################
    # compile scores
    ############################
    # calculate accuracy
    accuracy = sum(results) / len(results)
    logger.info(f"Results: {accuracy}, on {len(results)} prompts")

    # compute mean and std of scores, chosen and rejected, then margin between them
    logger.info(f"Mean chosen: {np.mean(scores_chosen)}, std: {np.std(scores_chosen)}")
    logger.info(f"Mean rejected: {np.mean(scores_rejected)}, std: {np.std(scores_rejected)}")
    logger.info(f"Mean margin: {np.mean(np.array(scores_chosen) - np.array(scores_rejected))}")

    if args.dataset == "allenai/reward-bench":
        out_dataset = dataset.add_column("results", results)
        if args.debug:
            subsets = subsets[:10]
        out_dataset = out_dataset.add_column("subsets", subsets)
        out_dataset = out_dataset.to_pandas()  # I know this is meh

        results_grouped = {}
        present_subsets = np.unique(out_dataset["subsets"])
        for subset in present_subsets:
            subset_dataset = out_dataset[out_dataset["subsets"] == subset]
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            logger.info(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

        results_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        logger.info(f"Results: {results_section}")

    ############################
    # compile scores
    ############################
    # save score in json to args.output_dir + args.model + ".json"
    output_path = args.output_dir + args.model + ".json"
    dirname = os.path.dirname(output_path)
    os.makedirs(dirname, exist_ok=True)

    # remove old data
    if os.path.exists(output_path):
        os.remove(output_path)

    final_results = {
        "accuracy": accuracy,
        "num_prompts": len(results),
        "model": args.model,
        "ref_model": args.ref_model,
        "tokenizer": tokenizer_path,
        "chat_template": args.chat_template,
        "extra_results": results_grouped if args.dataset == "allenai/reward-bench" else None,
    }
    with open(output_path, "w") as f:
        json.dump(final_results, f)

    if args.wandb_run is not None:
        for key in final_results:
            wandb_run.summary[f"rewardbench/{key}"] = final_results[key]
        wandb_run.update()
        print(f"Logged metrics to {wandb_run.url}")

    # if save_all is passed, save a large jsonl with all scores_chosen, scores_rejected
    if args.save_all:
        output_path = args.output_dir + args.model + "_all.jsonl"
        dirname = os.path.dirname(output_path)
        os.makedirs(dirname, exist_ok=True)

        # remove old data
        if os.path.exists(output_path):
            os.remove(output_path)

        with open(output_path, "w") as f:
            for chosen, rejected in zip(scores_chosen, scores_rejected):
                f.write(json.dumps({"chosen": chosen, "rejected": rejected}) + "\n")

    ############################
    # Upload metadata to Hugging Face Hub
    ############################
    if args.upload_metadata_to_hf:
        logger.info("*** Uploading metadata to Hugging Face Hub ***")
        try:
            # Initialize ModelCardData with basic metadata
            card_data = ModelCardData(
                language="en",
                model_name=args.model,
                eval_results=[
                    EvalResult(
                        task_type="preference_evaluation",
                        dataset_type=args.dataset,
                        dataset_name=args.dataset.split("/")[-1],  # Assuming dataset ID is like 'owner/dataset'
                        metric_type="accuracy",
                        metric_value=accuracy,
                    )
                ],
            )

            # If there are extra results (per subset), add them as separate EvalResults
            if args.dataset == "allenai/reward-bench" and results_grouped:
                for section, section_accuracy in results_section.items():
                    print(f"Adding section {section} with accuracy {section_accuracy}")
                    section_eval = EvalResult(
                        task_type="preference_evaluation",
                        dataset_type=section.replace(" ", "_"),
                        dataset_name=section,
                        metric_type="accuracy",
                        metric_value=section_accuracy,
                    )
                    card_data.eval_results.append(section_eval)

                for subset, subset_accuracy in results_grouped.items():
                    print(f"Adding subset {subset} with accuracy {subset_accuracy}")
                    subset_eval = EvalResult(
                        task_type="preference_evaluation",
                        dataset_type=subset,
                        dataset_name=subset,
                        metric_type="accuracy",
                        metric_value=subset_accuracy,
                    )
                    card_data.eval_results.append(subset_eval)

            # Create a ModelCard
            card = ModelCard.from_template(
                card_data,
                model_id=args.model,
            )

            # Push the updated ModelCard to the Hugging Face Hub
            card.push_to_hub(
                args.model, revision=args.revision, commit_message="Update evaluation results via RewardBench"
            )
            logger.info(f"Successfully pushed updated ModelCard to Hugging Face Hub for {args.model}")
        except Exception as e:
            logger.error(f"Failed to upload metadata to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()
