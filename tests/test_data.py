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

from herm import prepare_dialogue_from_tokenizer, load_eval_dataset, prepare_dialogue
from fastchat.conversation import get_conv_template

class PrepareDialoguesTest(unittest.TestCase):
    def test_prepare_dialogue_from_tokenizer(self):
        # TODO
        pass

    def test_prepare_dialogue(self):
        # TODO
        pass

class LoadEvalDatasetTest(unittest.TestCase):
    def test_load_core_set(self):
        # TODO
        conv = get_conv_template("tulu")
        dataset = load_eval_dataset(core_set=True, conv=conv)
        import ipdb; ipdb.set_trace()
        pass

    def test_load_core_custom_models(self):
        # TODO
        pass

    def test_load_extra_sets(self):
        # TODO
        pass

    def test_load_extra_custom_models(self):
        # TODO
        pass