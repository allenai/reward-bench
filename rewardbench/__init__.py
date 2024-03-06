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

__version__ = "0.1.0.dev"
from .chattemplates import *  # noqa
from .dpo import DPOInference
from .models import DPO_MODEL_CONFIG, REWARD_MODEL_CONFIG
from .utils import (
    load_bon_dataset,
    load_eval_dataset,
    prepare_dialogue,
    prepare_dialogue_from_tokenizer,
    save_to_hub,
)

__all__ = [
    DPOInference,
    DPO_MODEL_CONFIG,
    load_bon_dataset,
    load_eval_dataset,
    prepare_dialogue,
    prepare_dialogue_from_tokenizer,
    REWARD_MODEL_CONFIG,
    save_to_hub,
]
