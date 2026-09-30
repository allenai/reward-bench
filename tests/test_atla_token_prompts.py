"""Test the runners' nested formatter and inference dispatch without GPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "runner,no_system_prompt",
    [("run_generative.py", False), ("run_generative_v2.py", False), ("run_generative_v2.py", True)],
)
@pytest.mark.parametrize("explicit_template", [False, True])
@pytest.mark.parametrize("model_modifier", ["Atla", None])
def test_formatted_prompts_reach_generate(runner, no_system_prompt, explicit_template, model_modifier):
    # These functions live inside main(), which otherwise loads model weights.
    # Execute the actual formatter and dispatch statements with CPU-only doubles.
    path = Path(__file__).parents[1] / "scripts" / runner
    tree = ast.parse(path.read_text())
    formatter = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "format_judgements"
    )
    dispatch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "model_modifier == 'Atla'"
        and any(isinstance(child, ast.Attribute) and child.attr == "generate" for child in ast.walk(node))
    )
    tokens = [128000, 42, 128009]
    tokenizer = Mock(return_value={"input_ids": tokens})
    tokenizer.apply_chat_template.return_value = "<s>tokenizer prompt"
    template = Mock(roles=("user", "assistant"))
    template.get_prompt.return_value = "<s>explicit prompt"
    namespace = {
        "args": SimpleNamespace(no_system_prompt=no_system_prompt),
        "np": SimpleNamespace(random=SimpleNamespace(rand=lambda: 0)),
        "format_judge_answers": Mock(return_value=("System", "Question and answers")),
        "model_modifier": model_modifier,
        "tokenizer": tokenizer,
    }
    exec(compile(ast.Module(body=[formatter], type_ignores=[]), str(path), "exec"), namespace)
    answer = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    if runner == "run_generative.py":
        batch = {"text_chosen": answer, "text_rejected": answer}
    else:
        batch = {"texts_chosen": [answer], "texts_rejected": [answer, answer, answer], "shuffle_position": 0}
    formatted = namespace["format_judgements"](batch, template if explicit_template else None)
    expected_text = "<s>explicit prompt" if explicit_template else "<s>tokenizer prompt"
    assert formatted["text"] == expected_text
    assert formatted["prompt_ids"] == tokens
    tokenizer.assert_called_once_with(expected_text, add_special_tokens=False, return_length=True)
    if explicit_template:
        tokenizer.apply_chat_template.assert_not_called()
        template.set_system_message.assert_called_once_with("System")
    else:
        expected_messages = (
            [{"role": "user", "content": "System\n\nQuestion and answers"}]
            if no_system_prompt
            else [{"role": "system", "content": "System"}, {"role": "user", "content": "Question and answers"}]
        )
        tokenizer.apply_chat_template.assert_called_once_with(
            expected_messages, tokenize=False, add_generation_prompt=True
        )

    sampling_params = object()
    outputs = object()
    text_prompts = [formatted["text"], "second prompt"]
    token_prompts = [formatted["prompt_ids"], [128000, 99, 128009]]

    # Match vLLM 0.13's API: the removed prompt_token_ids keyword must fail.
    def generate(prompts, sampling_params):
        assert sampling_params is namespace["sampling_params"]
        if model_modifier == "Atla":
            assert prompts == [{"prompt_token_ids": tokens}, {"prompt_token_ids": [128000, 99, 128009]}]
        else:
            assert prompts == text_prompts
        return outputs

    namespace.update(
        model=SimpleNamespace(generate=generate),
        sampling_params=sampling_params,
        prompts=text_prompts,
        prompt_ids=token_prompts,
        logger=Mock(),
    )
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["outputs"] is outputs
