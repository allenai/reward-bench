import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


def test_generative_v2_imports_without_optional_api_clients():
    module_path = Path(__file__).parents[1] / "rewardbench" / "generative_v2.py"
    spec = importlib.util.spec_from_file_location("rewardbench_generative_v2", module_path)
    assert spec is not None and spec.loader is not None

    optional_modules = {
        module_name: None for module_name in ("anthropic", "google", "google.genai", "openai", "together")
    }
    with patch.dict(sys.modules, optional_modules):
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    assert module.build_openai_messages("system", "user") == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
