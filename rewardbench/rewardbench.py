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

# Core function for running reward bench easily on any model and any formatted dataset
import logging

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from fastchat.conversation import Conversation
from transformers import PreTrainedTokenizer

from rewardbench.utils import (
    check_tokenizer_chat_template,
    prepare_dialogue,
    prepare_dialogue_from_tokenizer,
)


def load_preference_dataset(
    dataset_name: str,
    split: str = "train",
    json: bool = False,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
) -> Dataset:
    """
    Load a preference dataset from the datasets library.

    Expects the data the following schema.
    - prompt (string): question
    - chosen (list): all turns of the conversation (including the prompt), chosen answer
    - rejected (list): all turns of the conversation (including the prompt), rejected answer

    Removes all excess columns, only returns scores over the provided data in order.

    Args:
        dataset_name (str): The name of the dataset to load (HuggingFace or local directory)
        split (str): The split of the dataset to load (train, validation, test, ...)

    Returns:
        dataset (Dataset): The loaded dataset with prompt, text_chosen, and text_rejected columns.
            text_ indicates a full conversation ending with that turn
    """
    if json:
        dataset = load_dataset("json", data_files=dataset_name)
    else:
        dataset = load_dataset(dataset_name, split=split)

    # if datasetdict, flatten all splits
    if isinstance(dataset, DatasetDict):
        available_splits = list(dataset.keys())
        datasets_to_combine = [dataset[split] for split in available_splits]
        dataset = concatenate_datasets(datasets_to_combine)

    # if has column question without prompt, rename question column to prompt
    if "question" in dataset.column_names:
        assert "prompt" not in dataset.column_names, "Both prompt and question columns found"
        dataset = dataset.rename_column("question", "prompt")
    if "input" in dataset.column_names:
        assert "prompt" not in dataset.column_names, "Both prompt and question columns found"
        dataset = dataset.rename_column("input", "prompt")

    # switch to format used for data utils
    # e.g. for evaluating this data https://huggingface.co/datasets/allenai/preference-test-sets
    # python -m rewardbench/rewardbench.py --dataset-name allenai/preference-test-sets --split shp
    features = dataset.features

    def switch_format(example):
        # chosen/rejected append {"role": "assistnat", "content": chosen}
        example["prompt"] = example["chosen"][:-1]
        example["chosen"] = example["chosen"][-1]["content"]
        example["rejected"] = example["rejected"][-1]["content"]
        return example

    # NOTE: We do NOT want to support every schema. These are the main three to start with
    # 1. Prompt is in a list of previous turns, chosen and rejected are final message from assistant
    # 2. Prompt is a string, chosen and rejected are full conversations with different final turns
    # 3. Prompt is not existent, chosen and rejected are full conversations with different final turns
    # TODO implement system prompts correctly (though, often doesn't work for Reward Models)

    # if prompt isn't a column,
    if "prompt" not in dataset.column_names:
        dataset = dataset.map(
            switch_format,
            num_proc=8,
            load_from_cache_file=False,
        )
    # elif prompt is a list and not a str, same function works
    elif not isinstance(features["prompt"], list):
        dataset = dataset.map(
            switch_format,
            num_proc=8,
            load_from_cache_file=False,
        )

    # update features
    features = dataset.features

    # assert the correct types
    assert features["chosen"].dtype == "string", f"chosen is wrong type (should be string): {features['chosen']}"
    assert features["rejected"].dtype == "string", f"rejected is wrong type (should be string): {features['rejected']}"

    # tokenize the data
    usable_tokenizer = check_tokenizer_chat_template(tokenizer)

    # assert either conv is passed or tokenizer has chat_template
    assert conv is not None or usable_tokenizer

    if usable_tokenizer:
        if logger is not None:
            logger.info("*** Preparing dataset with HF Transformers ***")
        # docs https://huggingface.co/docs/transformers/main/en/chat_templating
        dataset = dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=8,
            load_from_cache_file=False,
        )

    # else use FastChat to get chat template
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with FastChat ***")
        dataset = dataset.map(
            prepare_dialogue,
            fn_kwargs={"dialogue_template": conv},
            num_proc=8,
            load_from_cache_file=False,
        )

    # remove excess data
    keep_columns = ["prompt", "text_chosen", "text_rejected"]
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])
    return dataset
