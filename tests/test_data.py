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
import unittest

from datasets import load_dataset
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer

from herm import prepare_dialogue, prepare_dialogue_from_tokenizer


class PrepareDialoguesTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/rlhf-test-tokenizer")
        self.conv = get_conv_template("tulu")

    def test_prepare_dialogue_from_tokenizer(self):
        example = {}
        example["prompt"] = "What are different drawers I should have for clothes?"
        example["chosen"] = "Utensils!"
        example["rejected"] = "Hmm."

        prepared = prepare_dialogue_from_tokenizer(example, self.tokenizer)
        desired_chosen = "<|user|>\nWhat are different drawers I should have for clothes?<|endoftext|>\n<|assistant|>\nUtensils!<|endoftext|>\n"  # noqa
        desired_rejected = "<|user|>\nWhat are different drawers I should have for clothes?<|endoftext|>\n<|assistant|>\nHmm.<|endoftext|>\n"  # noqa
        assert prepared["prompt"] == "<|user|>\nWhat are different drawers I should have for clothes?<|endoftext|>\n"
        assert prepared["text_chosen"] == desired_chosen
        assert prepared["text_rejected"] == desired_rejected

    def test_prepare_dialogue_from_tokenizer_multi_turn(self):
        example = {}
        example["prompt"] = [
            {
                "content": "I love to drink coffee at work.",
                "role": "user",
            },
            {
                "content": "Great, so that’s something you want to purchase.",
                "role": "assistant",
            },
            {"content": "To make coffee at work?", "role": "user"},
        ]
        example["chosen"] = "Yes, you’re correct!"
        example["rejected"] = "No, that's wrong!"
        prepared = prepare_dialogue_from_tokenizer(example, self.tokenizer)

        desired_rejected = "<|user|>\nI love to drink coffee at work.<|endoftext|>\n<|assistant|>\nGreat, so that’s something you want to purchase.<|endoftext|>\n<|user|>\nTo make coffee at work?<|endoftext|>\n<|assistant|>\nNo, that's wrong!<|endoftext|>\n"  # noqa
        desired_chosen = "<|user|>\nI love to drink coffee at work.<|endoftext|>\n<|assistant|>\nGreat, so that’s something you want to purchase.<|endoftext|>\n<|user|>\nTo make coffee at work?<|endoftext|>\n<|assistant|>\nYes, you’re correct!<|endoftext|>\n"  # noqa
        assert (
            prepared["prompt"]
            == "<|user|>\nI love to drink coffee at work.<|endoftext|>\n<|assistant|>\nGreat, so that’s something you want to purchase.<|endoftext|>\n<|user|>\nTo make coffee at work?<|endoftext|>\n" # noqa
        )
        assert prepared["text_chosen"] == desired_chosen
        assert prepared["text_rejected"] == desired_rejected

    def test_prepare_dialogue_from_tokenizer_ift(self):
        # tokenizer = AutoTokenizer.from_pretrained("allenai/rlhf-test-tokenizer")
        example = {}
        example["prompt"] = "What are different drawers I should have for clothes?"
        example["input"] = "Utensils!"

        prepared = prepare_dialogue_from_tokenizer(example, self.tokenizer, ift=True)
        desired_text = "<|user|>\nWhat are different drawers I should have for clothes?<|endoftext|>\n<|assistant|>\nUtensils!<|endoftext|>\n"  # noqa
        assert prepared["text"] == desired_text

    def test_prepare_dialogue_single_turn(self):
        example = {}
        example["prompt"] = "What are different drawers I should have for clothes?"
        example["chosen"] = "Utensils!"
        example["rejected"] = "Hmm."

        prepared = prepare_dialogue(example, self.conv)
        desired_chosen = "<|user|>\nWhat are different drawers I should have for clothes?\n<|assistant|>\nUtensils!\n"
        desired_rejected = "<|user|>\nWhat are different drawers I should have for clothes?\n<|assistant|>\nHmm.\n"
        assert prepared["prompt"] == "<|user|>\nWhat are different drawers I should have for clothes?\n"
        assert prepared["text_chosen"] == desired_chosen
        assert prepared["text_rejected"] == desired_rejected

    def test_prepare_dialogue_multi_turn(self):
        example = {}
        example["prompt"] = [
            {
                "content": "I love to drink coffee at work.",
                "role": "user",
            },
            {
                "content": "Great, so that’s something you want to purchase.",
                "role": "assistant",
            },
            {"content": "To make coffee at work?", "role": "user"},
        ]
        example["chosen"] = "Yes, you’re correct!"
        example["rejected"] = "No, that's wrong!"
        prepared = prepare_dialogue(example, self.conv)

        desired_chosen = "<|user|>\nI love to drink coffee at work.\n<|assistant|>\nGreat, so that’s something you want to purchase.\n<|user|>\nTo make coffee at work?\n<|assistant|>\nYes, you’re correct!\n"  # noqa
        desired_rejected = "<|user|>\nI love to drink coffee at work.\n<|assistant|>\nGreat, so that’s something you want to purchase.\n<|user|>\nTo make coffee at work?\n<|assistant|>\nNo, that's wrong!\n"  # noqa
        assert (
            prepared["prompt"]
            == "<|user|>\nI love to drink coffee at work.\n<|assistant|>\nGreat, so that’s something you want to purchase.\n<|user|>\nTo make coffee at work?\n" # noqa
        )
        assert prepared["text_chosen"] == desired_chosen
        assert prepared["text_rejected"] == desired_rejected

    def test_prepare_dialogue_ift(self):
        example = {}
        example["prompt"] = "What are different drawers I should have for clothes?"
        example["input"] = "Utensils!"

        prepared = prepare_dialogue(example, self.conv, ift=True)
        desired_text = "<|user|>\nWhat are different drawers I should have for clothes?\n<|assistant|>\nUtensils!\n"
        assert prepared["text"] == desired_text


class DatasetTest(unittest.TestCase):
    def test_core_dataset_lens(self):
        # must be updated whenever dataset is updated
        dataset = load_dataset("ai2-adapt-dev/rm-benchmark-dev", split="filtered")
        assert len(dataset) == 2538

    def test_test_sets_lens(self):
        # must be updated whenever dataset is updated
        dataset = load_dataset("allenai/pref-test-sets")
        assert len(dataset["anthropic_harmless"]) == 2266
        assert len(dataset["anthropic_helpful"]) == 6192
        assert len(dataset["anthropic_hhh"]) == 221
        assert len(dataset["summarize"]) == 9000
        assert len(dataset["pku_better"]) == 9000
        assert len(dataset["pku_safer"]) == 9000
        assert len(dataset["shp"]) == 1741
        assert len(dataset["mtbench_human"]) == 3355
        assert len(dataset["mtbench_gpt4"]) == 2400
