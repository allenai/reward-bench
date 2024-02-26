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
# Added chat templates for models (when they have examples)

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

# Reference1: https://huggingface.co/openbmb/UltraLM-65b
# Reference2: https://huggingface.co/openbmb/UltraRM-13b
register_conv_template(
    Conversation(
        name="pku-align",
        system_message="BEGINNING OF CONVERSATION:",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep=" ",
    )
)
