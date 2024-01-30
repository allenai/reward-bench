#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Script to fine-tune a reward model on a Hub dataset

Adapted from:

- lvwerra/trl: https://github.com/lvwerra/trl/blob/main/examples/summarization/scripts/reward_summarization.py
- huggingface/transformers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
"""
import logging
import random
import subprocess
import sys
from datetime import timedelta

import datasets
import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

import evaluate
# import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from h4.data import get_datasets
from h4.training import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    get_dialogue_template,
    # init_wandb_training,
    prepare_dialogue,
    randargmax,
)
from h4.utils import (
    H4ArgumentParser,
    convert_to_safetensors,
    get_kbit_device_map,
    get_quantization_config,
    hf_login,
    is_adapter_model,
    is_slurm_available,
    push_to_hub_revision,
    run_rm_eval_job,
)
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training
from trl import RewardTrainer


logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16},"
        + f" training seet: {training_args.seed}, wandb enabled: {training_args.wandb_enabled}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    quit()

    # # Setup WandB
    # if training_args.wandb_enabled:
        # init_wandb_training(training_args)

    # Login to HuggingFace Hub if needed
    hf_login()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=6 * 1800))])

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    #####################################
    # Load tokenizer and process datasets
    #####################################
    # We truncate from the left to ensure we don't lose the human labels in the final dialogue turn
    tokenizer_kwargs = {
        "revision": model_args.model_revision,
        "use_fast": model_args.use_fast_tokenizer,
        "truncation_side": "left",
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        **tokenizer_kwargs,
    )

    # Preprocess the datasets
    with training_args.main_process_first(desc="Dataset map tokenization"):
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["test"].features)

        #########################
        # Apply dialogue template
        #########################
        dialogue_template = get_dialogue_template(data_args.dialogue_template)
        logger.info(f"System prompt for dialogue template: {dialogue_template.system}")
        raw_datasets = raw_datasets.map(
            prepare_dialogue,
            fn_kwargs={"dialogue_template": dialogue_template, "task": "rm"},
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Formatting comparisons with prompt template",
        )

        # Log a few random samples from the training set:
        if training_args.do_train:
            for index in random.sample(range(len(raw_datasets["train"])), 3):
                logger.info(
                    f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['text_chosen']}"
                )
                logger.info(
                    f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['text_rejected']}"
                )

        # Tokenize the dataset.
        def tokenize_function(examples):
            tokenized_chosen = tokenizer(
                examples["text_chosen"],
                max_length=data_args.max_source_length,
                truncation=False if data_args.filter_by_max_len else True,
            )
            tokenized_rejected = tokenizer(
                examples["text_rejected"],
                max_length=data_args.max_source_length,
                truncation=False if data_args.filter_by_max_len else True,
            )
            new_examples = {
                "input_ids_chosen": tokenized_chosen["input_ids"],
                "attention_mask_chosen": tokenized_chosen["attention_mask"],
                "input_ids_rejected": tokenized_rejected["input_ids"],
                "attention_mask_rejected": tokenized_rejected["attention_mask"],
            }
            return new_examples

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["text_chosen", "text_rejected"],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        if data_args.filter_by_max_len is True:
            logger.info(f"Filtering samples longer than {data_args.max_source_length} tokens")
            logger.info(f"Tokenized datasets before filtering: {tokenized_datasets}")
            tokenized_datasets = tokenized_datasets.filter(
                lambda x: len(x["input_ids_chosen"]) <= data_args.max_source_length
                and len(x["input_ids_rejected"]) <= data_args.max_source_length,
                num_proc=data_args.preprocessing_num_workers,
            )
            logger.info(f"Tokenized datasets after filtering: {tokenized_datasets}")

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    #######################
    # Load pretrained model
    #######################
    logger.info(f"*** Load pretrained model with {model_args.load_in_8bit=} and {model_args.load_in_4bit=} ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "num_labels": 1,
        "trust_remote_code": model_args.trust_remote_code,
        "use_cache": False,
    }
    if model_args.trust_remote_code and model_args.model_code_revision is not None:
        model_kwargs["code_revision"] = model_args.model_code_revision
    if model_args.use_peft is True:
        quantization_config = get_quantization_config(model_args)
        if is_adapter_model(model_args.model_name_or_path, revision=model_args.model_revision):
            logger.info("Loading adapter model")
            config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
            model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model_name_or_path,
                revision=model_args.model_code_revision,
                quantization_config=quantization_config,
                device_map=get_kbit_device_map(),
                **model_kwargs,
            )
        else:
            logger.info("Loading full base model")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                revision=model_args.model_revision,
                quantization_config=quantization_config,
                device_map=get_kbit_device_map(),
                **model_kwargs,
            )
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        model = get_peft_model(model, peft_config)
    else:
        logger.info(f"Loading base model with no adapters and {torch_dtype=}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            **model_kwargs,
        )

    logger.info(f"Setting tokenizer pad token to {tokenizer.eos_token}")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    ########################
    # Initialize the Trainer
    ########################
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # Here, predictions is rewards_chosen and rewards_rejected.
        # We want to see how much of the time rewards_chosen > rewards_rejected.
        # We use a random argmax for tie-breaking as naive argmax always gives the first (i.e. right) column, thus
        # giving spuriously high scores to degenerate models
        predictions = randargmax(predictions, axis=1)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        max_length=data_args.max_source_length,
    )

    ###############
    # Training loop
    ###############
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    trainer.save_model(training_args.output_dir)

    # # Save everything else on main process
    # if accelerator.is_main_process:
    #     kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    #     kwargs["dataset"] = list(data_args.dataset_mixer.keys())
    #     trainer.create_model_card(**kwargs)
    #     # Restore k,v cache for fast inference
    #     trainer.model.config.use_cache = True
    #     # Fix custom code paths
    #     if model_args.trust_remote_code is True:
    #         auto_map = trainer.model.config.auto_map
    #         trainer.model.config.auto_map = {k: v.split("--")[-1] for k, v in auto_map.items()}
    #     trainer.model.config.save_pretrained(training_args.output_dir)
    #     # Download custom modelling files
    #     if model_args.trust_remote_code:
    #         snapshot_download(
    #             repo_id=training_args.hub_model_id,
    #             local_dir=training_args.output_dir,
    #             allow_patterns="*.py",
    #         )
    #     # Store dialogue template so we can load it at deployment time
    #     dialogue_template.save_pretrained(training_args.output_dir)
    #     # FSDP/DeepSpeed save the model as a single `pytorch_model.bin` file, so we need to shard it.
    #     # We run this in a subprocess to avoid interference from the accelerators.
    #     subprocess.run(
    #         [
    #             "python",
    #             "scripts/training/shard_checkpoint.py",
    #             f"--output_dir={training_args.output_dir}",
    #             f"--trust_remote_code={model_args.trust_remote_code}",
    #         ],
    #         check=True,
    #     )
    #     # Convert torch weights to safetensors for deployment with TGI
    #     convert_to_safetensors(training_args.output_dir)
    #     if training_args.push_to_hub_revision:
    #         is_model_on_hub = push_to_hub_revision(training_args, model_args)
    #         # Run automatic evaluation once the model is pushed to the Hub
    #         if is_slurm_available() and is_model_on_hub is True and training_args.do_eval is True:
    #             logger.info("*** Launching automatic evaluation ***")
    #             run_rm_eval_job(training_args, model_args)

    accelerator.wait_for_everyone()
    # wandb.finish()

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
