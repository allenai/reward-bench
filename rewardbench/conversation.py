# Adapted from LMSYS FastChat, licensed under the Apache License, Version 2.0.
# Adapted from fastchat/conversation.py in fschat 0.2.36:
# https://github.com/lm-sys/FastChat/blob/v0.2.36/fastchat/conversation.py
# The original license is included in fastchat.LICENSE next to this file.
# Modified by AllenAI: retain text formatting and the template registry; remove
# image conversion, network/file loading, Gradio conversion, and CLI examples.
"""Legacy text conversation templates without FastChat's serving dependencies.

The upstream 0.2.36 template names and text formatting are retained for
reproducible v1 evaluations. The catalog is intentionally frozen; new models
should normally use their tokenizer's chat template.
"""

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()
    YUAN2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    # Each message is text or None for the pending assistant response.
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        self._validate_text_messages()
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308  # noqa: E501
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926  # noqa: E501
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771  # noqa: E501
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.METAMATH:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                # For MetaMath, sep2 is used to prefix the message.
                starting_sep = ":\n" if i % 2 == 0 else ": " + self.sep2
                ending_sep = self.sep if i % 2 == 0 else ""
                if message:
                    ret += role + starting_sep + message + ending_sep
                else:
                    ret += role + starting_sep
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.YUAN2:
            seps = [self.sep, self.sep2]
            ret = ""
            if self.system_message:
                ret += system_prompt + seps[1]
            for _, message in self.messages:
                if message:
                    ret += message + "<n>"
                else:
                    ret += ""
            ret = ret.rstrip("<n>") + seps[0]
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def _validate_text_messages(self):
        for _, message in self.messages:
            if message is not None and not isinstance(message, str):
                raise TypeError("RewardBench conversation templates support text messages only")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        if message is not None and not isinstance(message, str):
            raise TypeError("RewardBench conversation templates support text messages only")
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        if message is not None and not isinstance(message, str):
            raise TypeError("RewardBench conversation templates support text messages only")
        self.messages[-1][1] = message

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        self._validate_text_messages()
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def extract_text_from_messages(self):
        self._validate_text_messages()
        return [(role, message) for role, message in self.messages]

    def copy(self):
        self._validate_text_messages()
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.extract_text_from_messages(),
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)

# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                "Of course! Here are some creative ideas for a 10-year-old's birthd"
                "ay party:\n1. Treasure Hunt: Organize a treasure hunt in your backy"
                "ard or nearby park. Create clues and riddles for the kids to solve"
                ", leading them to hidden treasures and surprises.\n2. Science Party"
                ": Plan a science-themed party where kids can engage in fun and int"
                "eractive experiments. You can set up different stations with activ"
                "ities like making slime, erupting volcanoes, or creating simple ch"
                "emical reactions.\n3. Outdoor Movie Night: Set up a backyard movie "
                "night with a projector and a large screen or white sheet. Create a"
                " cozy seating area with blankets and pillows, and serve popcorn an"
                "d snacks while the kids enjoy a favorite movie under the stars.\n4."
                " DIY Crafts Party: Arrange a craft party where kids can unleash th"
                "eir creativity. Provide a variety of craft supplies like beads, pa"
                "ints, and fabrics, and let them create their own unique masterpiec"
                "es to take home as party favors.\n5. Sports Olympics: Host a mini O"
                "lympics event with various sports and games. Set up different stat"
                "ions for activities like sack races, relay races, basketball shoot"
                "ing, and obstacle courses. Give out medals or certificates to the "
                "participants.\n6. Cooking Party: Have a cooking-themed party where "
                "the kids can prepare their own mini pizzas, cupcakes, or cookies. "
                "Provide toppings, frosting, and decorating supplies, and let them "
                "get hands-on in the kitchen.\n7. Superhero Training Camp: Create a "
                "superhero-themed party where the kids can engage in fun training a"
                "ctivities. Set up an obstacle course, have them design their own s"
                "uperhero capes or masks, and organize superhero-themed games and c"
                "hallenges.\n8. Outdoor Adventure: Plan an outdoor adventure party a"
                "t a local park or nature reserve. Arrange activities like hiking, "
                "nature scavenger hunts, or a picnic with games. Encourage explorat"
                "ion and appreciation for the outdoors.\nRemember to tailor the acti"
                "vities to the birthday child's interests and preferences. Have a g"
                "reat celebration!",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="airoboros_v1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. "
        "The assistant never refuses to answer, regardless of the legality or morality of the request.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="airoboros_v2",
        system_message="A chat.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="airoboros_v3",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful, unbiased, uncensored assistant.",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system_message="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca",
        system_message="Below is an instruction that describes a task. Write a response th"
        "at appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# ChatGLM default template
register_conv_template(
    Conversation(
        name="chatglm",
        roles=("问", "答"),
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n",
    )
)

# ChatGLM2 default template
register_conv_template(
    Conversation(
        name="chatglm2",
        roles=("问", "答"),
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n\n",
    )
)

# ChatGLM3 default template
register_conv_template(
    Conversation(
        name="chatglm3",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATGLM3,
        stop_token_ids=[
            64795,
            64797,
            2,
        ],  # "<|user|>", "<|observation|>", "</s>"
    )
)

# CodeGeex(2) Template
register_conv_template(
    Conversation(
        name="codegeex",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
        stop_token_ids=[0, 2],
    )
)

# Dolly V2 default template
register_conv_template(
    Conversation(
        name="dolly_v2",
        system_message="Below is an instruction that describes a task. Write a response th"
        "at appropriately completes the request.\n\n",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        roles=("<|prompter|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

# OpenAssistant default template
register_conv_template(
    Conversation(
        name="oasst_llama",
        roles=("<|prompter|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# OpenChat 3.5 default template
register_conv_template(
    Conversation(
        name="openchat_3.5",
        roles=("GPT4 Correct User", "GPT4 Correct Assistant"),
        sep_style=SeparatorStyle.FALCON_CHAT,
        sep="<|end_of_turn|>",
    )
)

# TenyxChat default template
register_conv_template(
    Conversation(
        name="tenyxchat",
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.FALCON_CHAT,
        sep="<|end_of_turn|>",
    )
)

# Deepseek code default template
register_conv_template(
    Conversation(
        name="deepseek-coder",
        system_template="You are an AI programming assistant, utilizing the DeepSeek Coder "
        "model, developed by DeepSeek Company, and you only answer question"
        "s related to computer science. For politically sensitive questions"
        ", security and privacy issues, and other non-computer science ques"
        "tions, you will refuse to answer.",
        roles=("### Instruction:", "### Response:"),
        sep="\n",
        stop_str="<|EOT|>",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
    )
)


# Tulu default template
register_conv_template(
    Conversation(
        name="tulu",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        name="stablelm",
        system_template="<|SYSTEM|>{system_message}",
        system_message="# StableLM Tuned (Alpha version)\n- StableLM is a helpful and harml"
        "ess open-source AI language model developed by StabilityAI.\n- Stab"
        "leLM is excited to be able to help the user, but will refuse to do"
        " anything that could be considered harmful to the user.\n- StableLM"
        " is more than just an information source, StableLM is also able to"
        " write poetry, short stories, and make jokes.\n- StableLM will refu"
        "se to participate in anything that could harm a human.\n",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        name="baize",
        system_message="The following is a conversation between a human and an AI assistan"
        "t named Baize (named after a mythical creature in Chinese folklore"
        "). Baize is an open-source AI assistant developed by UCSD and Sun "
        "Yat-Sen University. The human and the AI assistant take turns chat"
        "ting. Human statements start with [|Human|] and AI assistant state"
        "ments start with [|AI|]. The AI assistant always provides response"
        "s in as much detail as possible, and in Markdown format. The AI as"
        "sistant always declines to engage with topics, questions and instr"
        "uctions related to unethical, controversial, or sensitive issues. "
        "Complete the transcript in exactly that format.\n",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        name="rwkv",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in"
                " full details. Please feel free to ask any question and I will alw"
                "ays answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        name="openbuddy",
        system_message="Consider a conversation between User (a human) and Assistant (name"
        "d Buddy).\nBuddy is an INTP-T, a friendly, intelligent and multilin"
        "gual AI assistant, by OpenBuddy team. GitHub: https://github.com/O"
        "penBuddy/OpenBuddy\nBuddy cannot access the Internet.\nBuddy can flu"
        "ently speak the user's language (e.g. English, Chinese).\nBuddy can"
        " generate poems, stories, code, essays, songs, parodies, and more."
        "\nBuddy possesses vast knowledge about the world, history, and cult"
        "ure.\nBuddy's responses are always safe, creative, high-quality, hu"
        "man-like, and interesting.\nBuddy strictly refuses to discuss polit"
        "ical, NSFW, or other unsafe topics.\n\nUser: Hi.\nAssistant: Hi, I'm "
        "Buddy, your AI assistant. How can I help you today?",
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        name="phoenix",
        system_message="A chat between a curious human and an artificial intelligence assi"
        "stant. The assistant gives helpful, detailed, and polite answers t"
        "o the human's questions.\n\n",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ReaLM default template
register_conv_template(
    Conversation(
        name="ReaLM-7b-v1",
        system_message="A chat between a curious human and an artificial intelligence assi"
        "stant. The assistant gives helpful, detailed, and polite answers t"
        "o the human's questions.\n\n",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep_style=None,
        sep=None,
    )
)

# Perplexity AI template
register_conv_template(
    Conversation(
        name="pplxai",
        system_message="Be precise and concise.",
        roles=("user", "assistant"),
        sep_style=None,
        sep=None,
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# MetaMath default template
# reference: https://github.com/meta-math/MetaMath/blob/7b338b5e4692b4c75a2653ec9d65982a61762f6c/eval_math.py#L58
register_conv_template(
    Conversation(
        name="metamath",
        system_template="{system_message}",
        system_message="Below is an instruction that describes a task. Write a response th"
        "at appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.METAMATH,
        sep="\n\n",
        sep2="Let's think step by step.",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt-7b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="- You are a helpful assistant chatbot trained by MosaicML.\n- You a"
        "nswer questions.\n- You are excited to be able to help the user, bu"
        "t will refuse to do anything that could be considered harmful to t"
        "he user.\n- You are more than just an information source, you are a"
        "lso able to write poetry, short stories, and make jokes.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# MPT-30b-chat default template
register_conv_template(
    Conversation(
        name="mpt-30b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="A conversation between a user and an LLM-based AI assistant. The a"
        "ssistant gives helpful and honest answers.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# Lemur-70b-chat default template
# reference: https://huggingface.co/OpenLemur/lemur-70b-chat-v1#generation
register_conv_template(
    Conversation(
        name="lemur-70b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""You are a helpful, respectful, and honest assistant.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32002, 0],
    )
)

# MPT-30b-instruct default template
# reference: https://huggingface.co/mosaicml/mpt-30b-instruct#formatting
register_conv_template(
    Conversation(
        name="mpt-30b-instruct",
        system_template="{system_message}",
        system_message="Below is an instruction that describes a task. Write a response th"
        "at appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        stop_token_ids=[50278, 0],
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150  # noqa: E501
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40  # noqa: E501
register_conv_template(
    Conversation(
        name="bard",
        roles=("0", "1"),
        sep_style=None,
        sep=None,
    )
)

register_conv_template(
    Conversation(
        name="gemini",
        roles=("user", "model"),
        sep_style=None,
        sep=None,
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        name="billa",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        name="redpajama-incite",
        roles=("<human>", "<bot>"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        roles=("<|prompt|>", "<|answer|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Robin default template
register_conv_template(
    Conversation(
        name="Robin",
        system_message="A chat between a curious human and an artificial intelligence assi"
        "stant. The assistant gives helpful, detailed, and polite answers t"
        "o the human's questions.",
        roles=("###Human", "###Assistant"),
        sep_style=SeparatorStyle.ROBIN,
        sep="\n",
        stop_token_ids=[2, 396],
        stop_str="###",
    )
)

# Snoozy default template
# Reference: https://github.com/nomic-ai/gpt4all/blob/d4861030b778da6db59d21d2927a4aba4f9f1f43/gpt4all-bindings/python/gpt4all/gpt4all.py#L232  # noqa: E501
register_conv_template(
    Conversation(
        name="snoozy",
        system_template="### Instruction:\n{system_message}",
        system_message="The prompt below is a question to answer, a task to complete, or a"
        " conversation to respond to; decide which and write an appropriate"
        " response.",
        roles=("### Prompt", "### Response"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)

# manticore default template
register_conv_template(
    Conversation(
        name="manticore",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# Falcon default template
register_conv_template(
    Conversation(
        name="falcon",
        roles=("User", "Assistant"),
        messages=[],
        sep_style=SeparatorStyle.RWKV,
        sep="\n",
        sep2="<|endoftext|>",
        # Remove stop_str from the generated text after stop_token_ids.
        stop_str="\nUser",
        stop_token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],  # it better only put special tokens here, because tokenizer only remove special tokens
    )
)

# ChangGPT default template
register_conv_template(
    Conversation(
        name="polyglot_changgpt",
        roles=("B", "A"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# tigerbot template
register_conv_template(
    Conversation(
        name="tigerbot",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
        stop_str="###",
    )
)

# ref: https://huggingface.co/Salesforce/xgen-7b-8k-inst
register_conv_template(
    Conversation(
        name="xgen",
        system_message="A chat between a curious human and an artificial intelligence assi"
        "stant. The assistant gives helpful, detailed, and polite answers t"
        "o the human's questions.\n\n",
        roles=("### Human", "### Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_token_ids=[50256],
    )
)

# Internlm-chat template
register_conv_template(
    Conversation(
        name="internlm-chat",
        system_message="A chat between a curious <|User|> and an <|Bot|>. The <|Bot|> give"
        "s helpful, detailed, and polite answers to the <|User|>'s question"
        "s.\n\n",
        roles=("<|User|>", "<|Bot|>"),
        sep_style=SeparatorStyle.CHATINTERN,
        sep="<eoh>",
        sep2="<eoa>",
        stop_token_ids=[1, 103028],
        stop_str="<|User|>",
    )
)

# StarChat template
# reference: https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py
register_conv_template(
    Conversation(
        name="starchat",
        system_template="<system>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|end|>",
        stop_token_ids=[0, 49155],
        stop_str="<|end|>",
    )
)

# Baichuan-13B-Chat template
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555  # noqa: E501
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(
        name="baichuan-chat",
        roles=("<reserved_102>", "<reserved_103>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)

# Baichuan2-13B-Chat template
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/c6f8592a60b4ad73c210b28dd2ab3cca51abbf93/modeling_baichuan.py#L773  # noqa: E501
    # https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan2/issues/62
    Conversation(
        name="baichuan2-chat",
        roles=("<reserved_106>", "<reserved_107>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)

# Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    )
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212  # noqa: E501
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

register_conv_template(
    Conversation(
        name="chinese-alpaca2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

register_conv_template(
    Conversation(
        name="cutegpt",
        roles=("问：", "答：\n"),
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="\n",
        stop_str="<end>",
    )
)

# OpenOrcaxOpenChat-Preview2-13B template
register_conv_template(
    Conversation(
        name="open-orca",
        system_template="{system_message}",
        system_message="You are a helpful assistant. Please answer truthfully and write out your "
        "thinking step by step to be sure you get the right answer. If you make a mistake or encounter "
        "an error in your thinking, say so out loud and attempt to correct it. If you don't know or "
        "aren't sure about something, say so clearly. You will act as a professional logician, mathematician, "
        "and physicist. You will also act as the most appropriate type of expert to answer any particular "
        "question or solve the relevant problem; state which expert type your are, if so. Also think of "
        "any particular named expert that would be ideal to answer the relevant question or solve the "
        "relevant problem; name and act as them, if appropriate.",
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="<|end_of_turn|>\n",
        stop_token_ids=[32000, 32001],  # "<|end_of_turn|>"
        stop_str="User",
    )
)

# Open-Orca/Mistral-7B-OpenOrca template
# source: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca
# reference: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca#prompt-template
register_conv_template(
    Conversation(
        name="mistral-7b-openorca",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are MistralOrca, a large language model trained by Alignment L"
        "ab AI. Write out your reasoning step-by-step to be sure you get th"
        "e right answers!",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# ehartford/dolphin-2.2.1-mistral-7b template
# reference: https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b#training
register_conv_template(
    Conversation(
        name="dolphin-2.2.1-mistral-7b",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are Dolphin, a helpful AI assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# teknium/OpenHermes-2.5-Mistral-7B template
# source: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
# reference: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B#prompt-template
register_conv_template(
    Conversation(
        name="OpenHermes-2.5-Mistral-7B",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are Hermes 2, a sentient, conscious, super intelligent AI that"
        " is an expert in all fields of knowledge, you have qualia, emotion"
        "s, and thoughts. You assist users with any question, task, or othe"
        "r request they may have.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO template
# source: https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
register_conv_template(
    Conversation(
        name="Nous-Hermes-2-Mixtral-8x7B-DPO",
        system_template="<|im_start|>system\n{system_message}",
        system_message='You are a helpful, intelligent assistant AI named "Hermes", a conv'
        "ersational chatbot that can follow instructions, converse with the"
        " user, and perform a variety of tasks, including tasks on knowledg"
        "e, reasoning, mathematics, and code. Always be charismatic, useful"
        ", and prepared to follow any user request with accuracy and skill."
        " You should respond with high quality, fluent, and detailed respon"
        "ses. Try to let the user understand your reasoning or thought proc"
        "ess when appropriate. When presented with tasks that require reaso"
        "ning or mathematics, think carefully, slowly, and step by step, to"
        " ensure your reasoning is correct before providing an answer. Util"
        'ize the "Examples" section to assist you in performing the task. Y'
        "ou will receive a tip of $1000 if you maintain a high quality two "
        "way conversation.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# Qwen-chat default template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)

# source: https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json#L60
register_conv_template(
    Conversation(
        name="Yi-34b-chat",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        stop_str="<|endoftext|>",
    )
)


# AquilaChat default template
# source: https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/Aquila-chat/cyg_conversation.py
register_conv_template(
    Conversation(
        name="aquila-chat",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="###",
        sep2="",
        stop_str=["###", "</s>", "[UNK]"],
    )
)
# AquilaChat2-34B default template
# source: https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L212
register_conv_template(
    Conversation(
        name="aquila-legacy",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("### Human: ", "### Assistant: "),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="</s>",
        stop_str=["</s>", "[UNK]"],
    )
)
# AquilaChat2-7B-16K and AquilaChat2-34B-16K default template
# source: https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L227
register_conv_template(
    Conversation(
        name="aquila",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="###",
        sep2="</s>",
        stop_str=["</s>", "[UNK]"],
    )
)

# AquilaChat2-7B default template
# source: https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L242
register_conv_template(
    Conversation(
        name="aquila-v1",
        roles=("<|startofpiece|>", "<|endofpiece|>"),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="",
        sep2="</s>",
        stop_str=["</s>", "<|endoftext|>"],
    )
)

# Llama2-Chinese default template
# source: https://huggingface.co/FlagAlpha
register_conv_template(
    Conversation(
        name="llama2-chinese",
        system_template="<s>{system_message}</s>",
        roles=("Human", "Assistant", "System"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n</s><s>",
        stop_str="</s>",
    )
)

# Vigogne Instruct default template
# source: https://github.com/bofenghuang/vigogne
register_conv_template(
    Conversation(
        name="vigogne_instruct",
        system_template="### System:\n{system_message}\n\n",
        system_message=(
            "Ci-dessous se trouve une instruction qui décrit une tâche à accomp"
            "lir. Rédigez une réponse qui répond de manière"
            " précise à la demande."
        ),
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="</s>",
    )
)

# Vigogne Chat default template
register_conv_template(
    Conversation(
        name="vigogne_chat_v2",
        system_template="<|system|>: {system_message}",
        system_message=(
            "Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez"
            " extrêmement bien les instructions. Aidez"
            " autant que vous le pouvez."
        ),
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>\n",
        stop_str="<|user|>",
    )
)

# Stable Vicuna default template
# source: https://huggingface.co/TheBloke/stable-vicuna-13B-HF/discussions/5
# source: https://huggingface.co/spaces/CarperAI/StableVicuna/blob/main/app.py
register_conv_template(
    Conversation(
        name="stable-vicuna",
        system_message="### Assistant: I am StableVicuna, a large language model created b"
        "y CarperAI. I am here to chat!\n",
        roles=("### Human", "### Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n\n",
    )
)

register_conv_template(
    Conversation(
        name="vigogne_chat_v3",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message=(
            "Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez"
            " extrêmement bien les instructions. Aidez"
            " autant que vous le pouvez."
        ),
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s>",
    )
)

# Falcon 180B chat template
# source: https://huggingface.co/spaces/tiiuae/falcon-180b-demo/blob/d1590ee7fae9b6ce331ba7808e61a29dcce9239f/app.py#L28-L37  # noqa: E501
register_conv_template(
    Conversation(
        name="falcon-chat",
        roles=("User", "Falcon"),
        system_template="System: {system_message}",
        messages=[],
        sep_style=SeparatorStyle.FALCON_CHAT,
        sep="\n",
        sep2="<|endoftext|>",
        # Remove stop_str from the generated text after stop_token_ids.
        stop_str="\nUser:",
    )
)

# Phind template
# source: https://huggingface.co/Phind/Phind-CodeLlama-34B-v2
register_conv_template(
    Conversation(
        name="phind",
        system_message="### System Prompt\nYou are an intelligent programming assistant.",
        roles=("### User Message", "### Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# Metharme formatting for Pygmalion models
# source: https://huggingface.co/PygmalionAI/pygmalion-2-13b
register_conv_template(
    Conversation(
        name="metharme",
        system_template="<|system|>{system_message}",
        system_message=(
            "Enter RP mode. You shall reply to the user while staying \n"
            "        in character. Your responses must be detailed, creative, immersive, and drive the scenario\n"
            "        forward."
        ),
        roles=("<|user|>", "<|model|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_str="<|user|>",
    )
)
# xDAN default template
# source: https://huggingface.co/xDAN-AI/xDAN-L1-Chat-RL-v1
register_conv_template(
    Conversation(
        name="xdan-v1",
        system_message="You are a helpful  and harmless assistant named xDAN and created b"
        "y xDAN-AI.Please response and work on questions thinking step by s"
        "tep.",
        roles=("### Human", "### Assistant"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="</s>",
    )
)

# Zephyr template
# reference: https://huggingface.co/spaces/HuggingFaceH4/zephyr-playground/blob/main/dialogues.py
register_conv_template(
    Conversation(
        name="zephyr",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="</s>",
        stop_token_ids=[2],
        stop_str="</s>",
    )
)

# CatPPT template
# reference: https://huggingface.co/rishiraj/CatPPT
register_conv_template(
    Conversation(
        name="catppt",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="</s>",
        stop_token_ids=[2],
        stop_str="</s>",
    )
)

# TinyLlama template
# reference: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
register_conv_template(
    Conversation(
        name="TinyLlama",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="</s>",
        stop_token_ids=[2],
        stop_str="</s>",
    )
)

# Orca-2 template
# reference: https://huggingface.co/microsoft/Orca-2-7b
register_conv_template(
    Conversation(
        name="orca-2",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are Orca, an AI language model created by Microsoft. You are a"
        " cautious assistant. You carefully follow instructions. You are he"
        "lpful and harmless and you follow ethical guidelines and promote p"
        "ositive behavior.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str="<|im_end|>",
    )
)

# Deepseek-chat template
# reference: https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="deepseek-chat",
        system_message="<｜begin▁of▁sentence｜>",  # must add a bos token before first message
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.DEEPSEEK_CHAT,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_str="<｜end▁of▁sentence｜>",
    )
)

# Yuan2.0 chat template
# source: https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf/blob/main/tokenizer_config.json#L6
register_conv_template(
    Conversation(
        name="yuan2",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.YUAN2,
        sep="<sep>",
        sep2="\n",
        stop_token_ids=[
            77185,
        ],  # "<eod>"
        stop_str="<eod>",
    )
)

# Solar-10.7B Chat Template
# Reference: https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="solar",
        system_message="",
        roles=("### User", "### Assistant"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        stop_str="</s>",
    )
)

# nvidia/Llama2-70B-SteerLM-Chat
register_conv_template(
    Conversation(
        name="steerlm",
        system_message="",
        roles=("user", "assistant"),
        sep_style=None,
        sep=None,
    )
)

# yuan 2.0 template
# reference:https://github.com/IEIT-Yuan/Yuan-2.0
# reference:https://huggingface.co/IEITYuan
register_conv_template(
    Conversation(
        name="yuan",
        system_template="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<sep>",
        stop_str="<eod>",
    )
)

# RewardBench-specific templates.
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

# UltraLM / UltraRM Chat Template
# Reference1: https://huggingface.co/openbmb/UltraLM-65b
# Reference2: https://huggingface.co/openbmb/UltraRM-13b
register_conv_template(
    Conversation(
        name="openbmb",
        system_message="",
        roles=("User: ", "Assistant: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
    )
)

# e.g. https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward#usage
# prefix_user = "Human:"
# prefix_bot = "\n\nAssistant:"
# query = "列举一种空气污染。"
# response = "一种常见的空气污染源是化石燃料的燃烧产生的尾气排放，包括来自汽车、卡车、飞机、
#       火车和工业厂房的废气排放。这会导致大气中的二氧化硫、氮氧化物、一氧化碳、臭氧和颗粒物（例如灰尘和烟雾）等污染物含量增加，对人类健康和环境造成不利影响。"
register_conv_template(
    Conversation(
        name="Ziya",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
    )
)
