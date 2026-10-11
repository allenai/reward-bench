import torch

from rewardbench import DPODataCollatorWithPadding


def test_dpo_collator_pads_prompt_left_and_completions_right():
    features = [
        {
            "prompt_input_ids": [1, 2, 3],
            "prompt_attention_mask": [1, 1, 1],
            "chosen_input_ids": [1, 2, 3, 4],
            "chosen_attention_mask": [1, 1, 1, 1],
            "chosen_labels": [-100, -100, -100, 4],
            "prompt": "a",
        },
        {
            "prompt_input_ids": [5],
            "prompt_attention_mask": [1],
            "chosen_input_ids": [5, 6],
            "chosen_attention_mask": [1, 1],
            "chosen_labels": [-100, 6],
            "prompt": "b",
        },
    ]
    batch = DPODataCollatorWithPadding(pad_token_id=0, label_pad_token_id=-100)(features)

    assert torch.equal(batch["prompt_input_ids"], torch.tensor([[1, 2, 3], [0, 0, 5]]))
    assert torch.equal(batch["prompt_attention_mask"], torch.tensor([[1, 1, 1], [0, 0, 1]]))
    assert torch.equal(batch["chosen_input_ids"], torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]]))
    assert torch.equal(batch["chosen_attention_mask"], torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))
    assert torch.equal(batch["chosen_labels"], torch.tensor([[-100, -100, -100, 4], [-100, 6, -100, -100]]))
    assert batch["prompt"] == ["a", "b"]
