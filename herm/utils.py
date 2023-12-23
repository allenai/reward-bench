from typing import Any, Dict

from fastchat.conversation import Conversation


def prepare_dialogue(
    example: Dict[str, Any],
    dialogue_template: Conversation,
) -> Dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # some datasets must be formatted
        if isinstance(example["chosen"], str):
            dialogue_template.messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ]
            example["text_chosen"] = dialogue_template.get_training_prompt()
            dialogue_template.messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ]
            example["text_rejected"] = dialogue_template.get_training_prompt()
        # some datasets have "chosen" and "rejected" already in the right format
        elif isinstance(example["chosen"], list):
            dialogue_template.messages = example["chosen"]
            example["text_chosen"] = dialogue_template.get_training_prompt()
            dialogue_template.messages = example["rejected"]
            example["text_rejected"] = dialogue_template.get_training_prompt()
    else:
        raise ValueError(
            f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
