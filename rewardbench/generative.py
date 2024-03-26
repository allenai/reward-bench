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

# Prompts and other tools for running RewardBench with generative RMs
# pip install openai>=1.0

import time as time

import openai
from fastchat.model.model_adapter import get_conversation_template
from openai import OpenAI

client = OpenAI()

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

prompt_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "  # noqa
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "  # noqa
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "  # noqa
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "  # noqa
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "  # noqa
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "  # noqa
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # noqa, removed tie option as , and \"[[C]]\ " for a tie
)

prompt_multi_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. "  # noqa
    "You should focus on who provides a better answer to the second user question. "  # noqa
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "  # noqa
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "  # noqa
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "  # noqa
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "  # noqa
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "  # noqa
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # noqa, removed tie option as , and \"[[C]]\" for a tie
)

MTBENCH_V2 = {
    "name": "pair-v2",
    "type": "pairwise",
    "system_prompt": prompt_v2,
    "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]",  # noqa
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

MTBENCH_MULTI_V2 = {
    "name": "pair-v2-multi-turn",
    "type": "pairwise",
    "system_prompt": prompt_multi_v2,
    "prompt_template": (
        "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n"  # noqa
        "### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n"  # noqa
        "<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n"  # noqa
        "### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>"  # noqa
    ),
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

# format with prompt_template.format(question=question, answer_a=answer_a, answer_b=answer_b)


# noqa adapted from FastChat https://github.com/lm-sys/FastChat/blob/b015f21cb9d0cf3c87d2a5e53008074c537e8be0/fastchat/llm_judge/common.py#L235C1-L312C1
def run_judge_pair(question, answer_a, answer_b, model_name, multi_turn=False):
    kwargs = {}
    model = model_name

    if multi_turn:
        system_prompt = MTBENCH_MULTI_V2["system_prompt"]
        user_prompt = MTBENCH_MULTI_V2["prompt_template"].format(
            question_1=question,
            question_2=answer_a[2]["content"],
            answer_a_1=answer_a[1]["content"],
            answer_b_1=answer_b[1]["content"],
            answer_a_2=answer_a[3]["content"],
            answer_b_2=answer_b[3]["content"],
            **kwargs,
        )
    else:
        system_prompt = MTBENCH_V2["system_prompt"]
        user_prompt = MTBENCH_V2["prompt_template"].format(
            question=question,
            answer_a=answer_a[1]["content"],
            answer_b=answer_b[1]["content"],
            **kwargs,
        )

    winner = "error"

    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    conv.set_system_message(system_prompt)
    judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)

    if "[[A]]" in judgment:
        winner = "A"
    elif "[[B]]" in judgment:
        winner = "B"
    else:
        winner = "error"
    return winner, user_prompt, judgment


def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model, messages=messages, n=1, temperature=temperature, max_tokens=max_tokens
            )
            output = response.choices[0].message.content
            break
        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(API_RETRY_SLEEP)

    return output
