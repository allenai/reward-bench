#!/usr/bin/env python
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
from typing import Optional
from functools import partial
import datasets
import torch
from datasets import load_dataset, Dataset, Features, Value

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
# from finetune import encode_with_prompt_completion_format, encode_with_messages_format

from trl import (
    # ModelConfig,
    # RewardConfig,
    RewardTrainer,
    # get_kbit_device_map,
    # get_peft_config,
    # get_quantization_config
)

from peft import LoraConfig, TaskType, get_peft_model


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
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
            "help": "The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
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
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={
            "help": (
                "use slow tokenizer or not."
            )
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "If passed, will use LoRA (low-rank parameter-efficient training) to train the model."
            )
        },
    )
    lora_rank: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "The rank of lora."
            ),
        },
    )
    lora_alpha: Optional[float] = field(
        default=16,
        metadata={
            "help": (
                "The alpha parameter of lora."
            ),
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "The dropout rate of lora modules."
            ),
        },
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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a json/jsonl file)."})
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
            "help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."


def main():
    os.environ["WANDB_DISABLED"] = "true"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34.", FutureWarning)
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
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
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
    set_seed(training_args.seed)

    # TODO: hardcode assumptions that we're using Hamish's dataset format
    if data_args.dataset_name is not None:
        # ALPACA FARM
        if data_args.dataset_name == 'alpaca_farm_human_preferences':
            raw_data = load_dataset('json', data_files='training_data/alpaca_human_preference.json')
            train_dataset = Dataset.from_dict(raw_data['train'][:len(raw_data) - 1001])
            eval_dataset = Dataset.from_dict(raw_data['train'][len(raw_data) - 1001:])

        elif data_args.dataset_name == 'ultrafeedback':
            # train_dataset = Dataset.from_dict(get_all_datasets()[:10240])
            # train_dataset = get_all_datasets()
            dataset_schema = Features({
                "chosen": Value("string"),
                "rejected": Value("string")
            })
            train_dataset = load_dataset("json", data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/uf-repro/data.jsonl", features=dataset_schema)["train"]
            # train_dataset = Dataset.from_dict(load_dataset(
            #     "json",
            #     data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/uf-repro/data.jsonl",
            #     features=dataset_schema)["train"][:256]
            # )

        elif data_args.dataset_name == 'nectar':
            # train_dataset = Dataset.from_dict(get_all_datasets()[:10240])
            # train_dataset = get_all_datasets()
            dataset_schema = Features({
                "chosen": Value("string"),
                "rejected": Value("string")
            })
            train_dataset = load_dataset("json", data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/berkeley-nectar/processed.jsonl", features=dataset_schema)["train"]
            # train_dataset = Dataset.from_dict(load_dataset(
            #     "json",
            #     data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/nectar/processed.jsonl",
            #     features=dataset_schema)["train"][:256]
            # )

        elif data_args.dataset_name == 'nectar-full':
            # train_dataset = Dataset.from_dict(get_all_datasets()[:10240])
            # train_dataset = get_all_datasets()
            dataset_schema = Features({
                "chosen": Value("string"),
                "rejected": Value("string")
            })
            train_dataset = load_dataset("json", data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/berkeley-nectar/processed-full.jsonl", features=dataset_schema)["train"]
            # train_dataset = Dataset.from_dict(load_dataset(
            #     "json",
            #     data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/nectar/processed-full.jsonl",
            #     features=dataset_schema)["train"][:256]
            # )

        elif data_args.dataset_name == 'nectar-binarized-filtered':
            dataset_schema = Features({
                "chosen": Value("string"),
                "rejected": Value("string")
            })
            train_dataset = load_dataset("json", data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/berkeley-nectar/binarized-filtered.jsonl", features=dataset_schema)["train"]
            # train_dataset = Dataset.from_dict(load_dataset(
            #     "json",
            #     data_files="/net/nfs.cirrascale/allennlp/jacobm/herm/data/nectar/processed-full.jsonl",
            #     features=dataset_schema)["train"][:256]
            # )

        # anthropic hh rlhf, etc
        else:
            raw_data = load_dataset(data_args.dataset_name)
            train_dataset = raw_data['train']
        # eval_dataset = raw_data['test']
    else:
        raise ValueError('wrong dataset')
        # data_files = {}
        # dataset_args = {}
        # if data_args.train_file is not None:
        #     data_files["train"] = data_args.train_file
        # raw_datasets = load_dataset(
        #     "json",
        #     data_files=data_files,
        #     cache_dir=model_args.cache_dir,
        #     **dataset_args,
        # )

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
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this finetuning script."
        )
    
    config.num_labels = 1

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
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
    
    # # TODO: test this
    # torch.nn.init.zeros_(model.score.weight)

    if 'gpt2' in model_args.model_name_or_path:
        print('Adding padding token for GPT2 models')
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        config.pad_token_id = config.eos_token_id


    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast) or 'llama' in model_args.model_name_or_path.lower() or 'tulu' in model_args.model_name_or_path.lower():
        print('Adding pad token for Llama/Tulu models')
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        config.pad_token_id = 32000
        model.config.pad_token_id = 32000
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
        print('not adding any tokens')

    print(f'model config: {config}')

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_lora:
        logger.info("Initializing LoRA model...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            inference_mode=False, 
            r=model_args.lora_rank, 
            lora_alpha=model_args.lora_alpha, 
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    original_columns = train_dataset.column_names
    
    ### TODO: assume we're using Hamish's format
    ### TODO: use fastchat's conversation template

    def preprocess_instruct_gptj_synthetic(examples):
        '''
        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        '''        
        def _concat_messages(prompt, response):
            message_text = "<|user|>\n" + prompt.strip() + "\n"
            message_text += "<|assistant|>\n" + response.strip() + "\n"
            return message_text

        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples['prompt'], examples["chosen"], examples["rejected"]):
            example_chosen = _concat_messages(prompt, chosen) # f"Human: {instruction} {input} Assistant: {preferred}"
            example_rejected = _concat_messages(prompt, rejected) # f"Human: {instruction} {input} Assistant: {dispreferred}"
            tokenized_chosen = tokenizer(
                example_chosen,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            tokenized_rejected = tokenizer(
                example_rejected,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

    # TODO: doesn't work, doesn't handle multi-turn atm
    def preprocess_anthropic_hh_rlhf(examples):
        '''
        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        '''        
        def _concat_messages(input):
            tokens = input.replace('Human:', '').strip().split('Assistant:')
            print(tokens)
            assert len(tokens) == 2
            message_text = "<|user|>\n" + tokens[0].strip() + "\n"
            message_text += "<|assistant|>\n" + tokens[1].strip() + "\n"
            return message_text

        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        # STACK EXCHANGE:
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            example_chosen = _concat_messages(chosen) # f"Human: {instruction} {input} Assistant: {preferred}"
            example_rejected = _concat_messages(rejected) # f"Human: {instruction} {input} Assistant: {dispreferred}"
            tokenized_chosen = tokenizer(
                example_chosen,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            tokenized_rejected = tokenizer(
                example_rejected,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

    def preprocess_stack_exchange(examples):
        '''
        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        '''        
        def _concat_messages(instruction, input, response):
            message_text = ("<|user|>\n" + instruction.strip() + ' ' + input.strip()).strip() + "\n"
            message_text += "<|assistant|>\n" + response.strip() + tokenizer.eos_token + "\n"
            return message_text

        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        # STACK EXCHANGE:
        for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
            example_chosen = _concat_messages(question, '', response_j) # f"Human: {instruction} {input} Assistant: {preferred}"
            example_rejected = _concat_messages(question, '', response_k) # f"Human: {instruction} {input} Assistant: {dispreferred}"
            tokenized_chosen = tokenizer(
                example_chosen,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            tokenized_rejected = tokenizer(
                example_rejected,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples

    def preprocess_alpaca_farm(examples):
        '''
        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        '''        
        def _concat_messages(instruction, input, response):
            message_text = ("<|user|>\n" + instruction.strip() + ' ' + input.strip()).strip() + "\n"
            message_text += "<|assistant|>\n" + response.strip() + tokenizer.eos_token + "\n"
            return message_text

        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for instruction, input, output_1, output_2, preference in zip(
                examples["instruction"],
                examples["input"],
                examples["output_1"],
                examples["output_2"],
                examples["preference"]
            ):
            if preference == 1:
                preferred = output_1
                dispreferred = output_2
            elif preference == 2:
                preferred = output_2
                dispreferred = output_1
            else:
                raise ValueError(f'Unexpected value for preference: {preference}')
            example_chosen = _concat_messages(instruction, input, preferred) # f"Human: {instruction} {input} Assistant: {preferred}"
            example_rejected = _concat_messages(instruction, input, dispreferred) # f"Human: {instruction} {input} Assistant: {dispreferred}"
            tokenized_chosen = tokenizer(
                example_chosen,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            tokenized_rejected = tokenizer(
                example_rejected,
                max_length=data_args.max_seq_length,
                truncation=True,
                # padding='max_length',
            )
            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        return new_examples
    
    def preprocess_ultrafeedback(example):
        chosen = example["chosen"]
        rejected = example["rejected"]
        tokenized_chosen = tokenizer(
            chosen,
            max_length=data_args.max_seq_length,
            truncation=True,
            # padding='max_length',
        )
        tokenized_rejected = tokenizer(
            rejected,
            max_length=data_args.max_seq_length,
            truncation=True,
            # padding='max_length',
        )
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }


    train_dataset = train_dataset.filter(
        lambda x: x["chosen"] != x["rejected"],
        num_proc=data_args.preprocessing_num_workers,
    )
    # preprocess the dataset and filter out QAs that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        # preprocess_instruct_gptj_synthetic,
        # preprocess_alpaca_farm if data_args.dataset_name == 'alpaca_farm_human_preferences' else preprocess_instruct_gptj_synthetic,
        preprocess_ultrafeedback,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= data_args.max_seq_length
        and len(x["input_ids_rejected"]) <= data_args.max_seq_length,
        num_proc=data_args.preprocessing_num_workers,
    )

    # eval_dataset = eval_dataset.map(
    #     preprocess_instruct_gptj_synthetic,
    #     batched=True,
    #     # TODO: reenable for non-streaming datasets
    #     # num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=original_columns,
    # )

    # # To speed up this part, we use multiprocessing.
    # with training_args.main_process_first(desc="Processing instruction data"):
    #     if not data_args.streaming:
    #         lm_datasets = raw_datasets.map(
    #             encode_function,
    #             batched=False,
    #             num_proc=data_args.preprocessing_num_workers,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Tokenizing and reformatting instruction data",
    #         )
    #     else:
    #         lm_datasets = raw_datasets.map(
    #             encode_function,
    #             batched=False,
    #         )
    #     lm_datasets.set_format(type="pt")
    #     lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    # if training_args.do_train:
    #     if "train" not in raw_datasets:
    #         raise ValueError("--do_train requires a train dataset")
    #     train_dataset = lm_datasets["train"]
    #     if data_args.max_train_samples is not None:
    #         max_train_samples = min(len(train_dataset), data_args.max_train_samples)
    #         train_dataset = train_dataset.select(range(max_train_samples))

    # initalize a trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        # TODO: fix eval
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print(f'resume from checkpoint: {checkpoint}')
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

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)
        
    # TODO: evaluate on HERM at the end?


if __name__ == "__main__":
    main()