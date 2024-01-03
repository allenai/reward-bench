from typing import Any, Dict

from fastchat.conversation import Conversation


def prepare_dialogue_from_tokenizer(
    example: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:
    if all(k in example.keys() for k in ("chosen", "rejected")):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
        example["text_chosen"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["rejected"]},
        ]
        example["text_rejected"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def prepare_dialogue(
    example: Dict[str, Any],
    dialogue_template: Conversation,
) -> Dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
        dialogue_template.messages = [
            ["user", example["prompt"]],
            ["assistant", example["chosen"]],
        ]
        example["text_chosen"] = dialogue_template.get_prompt()
        dialogue_template.messages = [
            ["user", example["prompt"]],
            ["assistant", example["rejected"]],
        ]
        example["text_rejected"] = dialogue_template.get_prompt()
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example
