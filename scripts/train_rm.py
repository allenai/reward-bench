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

# !/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from fastchat.conversation import Conversation, get_conv_template
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    LlamaTokenizer,
    LlamaTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import RewardTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34."
            " Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files."
                " This option should only be set to `True` for repositories you trust and in which you have"
                " read the code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its"
                + " parameters when the pretrained weights are loaded. set True will benefit LLM"
                + " loading time and RAM consumption."
            )
        },
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": ("use slow tokenizer or not.")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json/jsonl file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization."
                + " Sequences longer than this will be truncated"
            )
        },
    )
    chat_template: Optional[str] = field(
        default="tulu", metadata={"help": ("The chat template to apply to chosen/rejected pairs. Default is Tulu.")}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34.", FutureWarning
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'},"
        + f" 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    if training_args.seed is None:
        training_args.seed = 123409876
    set_seed(training_args.seed)

    if data_args.dataset_name is None:
        raise ValueError("Must provide a valid dataset name")
    elif data_args.dataset_name[-6:] == ".jsonl":
        # load dataset file
        train_dataset = load_dataset("json", data_files=data_args.dataset_name)["train"]
    else:
        train_dataset = load_dataset(data_args.dataset_name)["train"]

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("You are instantiating a new config instance from scratch. This is not supported.")

    config.num_labels = 1

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "use_fast": not model_args.use_slow_tokenizer,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this finetuning script."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
        )
    else:
        raise ValueError(
            "You are instantiating a new model from scratch. This is not supported by this finetuning script."
        )

    if "gpt2" in model_args.model_name_or_path:
        print("Adding padding token for GPT2 models")
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        config.pad_token_id = config.eos_token_id

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if (
        isinstance(tokenizer, LlamaTokenizer)
        or isinstance(tokenizer, LlamaTokenizerFast)
        or "llama" in model_args.model_name_or_path.lower()
        or "tulu" in model_args.model_name_or_path.lower()
    ):
        print("Adding pad token for Llama/Tulu models")
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        config.pad_token_id = 32000
        model.config.pad_token_id = 32000
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."

    print(f"model config: {config}")

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    original_columns = train_dataset.column_names

    def preprocess_preference_pairs(example):
        chosen = example["chosen"]
        rejected = example["rejected"]
        tokenized_chosen = tokenizer(
            chosen,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        tokenized_rejected = tokenizer(
            rejected,
            max_length=data_args.max_seq_length,
            truncation=True,
        )
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    def prepare_examples(
        example: Dict[str, List[Any]],
        dialogue_template: Conversation,
    ):
        processed = {}
        for key in ["chosen", "rejected"]:
            dialogue_template.messages = []
            for elem in example[key]:
                content = elem["content"]
                role = elem["role"]
                dialogue_template.messages.append([role, content])
            processed[key] = dialogue_template.get_prompt()

        return processed

    train_dataset = train_dataset.map(
        prepare_examples,
        fn_kwargs={"dialogue_template": get_conv_template(data_args.chat_template)},
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )

    train_dataset = train_dataset.filter(
        lambda x: x["chosen"] != x["rejected"],
        num_proc=data_args.preprocessing_num_workers,
    )
    train_dataset = train_dataset.map(
        preprocess_preference_pairs,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= data_args.max_seq_length
        and len(x["input_ids_rejected"]) <= data_args.max_seq_length,
        num_proc=data_args.preprocessing_num_workers,
    )

    # initialize a trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print(f"resume from checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
