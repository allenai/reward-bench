"""Protocol metadata must not become a score in analysis consumers."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


def load_analysis_module(name):
    path = Path(__file__).parents[1] / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_analysis_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("include_protocol", [False, True])
def test_result_averages_ignore_protocol_metadata(tmp_path, monkeypatch, include_protocol):
    utils = load_analysis_module("utils")
    load_dataset = utils.load_dataset
    monkeypatch.setattr(
        utils, "load_dataset", lambda *args, **kwargs: load_dataset(*args, cache_dir=str(tmp_path / "cache"), **kwargs)
    )
    result_dir = tmp_path / "eval-set" / "example"
    result_dir.mkdir(parents=True)
    for name, protocol in [("legacy", None), ("ranking", False), ("ratings", True)]:
        payload = {
            "model": f"example/{name}",
            "model_type": "Generative RM",
            "chat_template": None,
            "Factuality": 0.2,
            "Reasoning": 0.8,
        }
        if include_protocol and protocol is not None:
            payload["score_w_ratings"] = protocol
        (result_dir / f"{name}.json").write_text(json.dumps(payload))

    results = utils.load_results(tmp_path, "eval-set")
    assert len(results) == 3
    np.testing.assert_allclose(results["average"].to_numpy(dtype=float), [0.5, 0.5, 0.5])
    assert "score_w_ratings" not in results.columns
    assert set(results.columns) == {"model", "model_type", "average", "Factuality", "Reasoning"}


def test_subset_plot_ignores_mixed_boolean_and_missing_protocol(monkeypatch):
    # Mock only rendering/imports: execute the real numeric plot preparation
    # without requiring matplotlib, model dependencies, or network downloads.
    axes = [Mock(spines={"right": Mock(), "top": Mock()}) for _ in range(2)]
    for axis in axes:
        axis.violinplot.return_value = {"bodies": [Mock()]}
    plt = Mock(rcParams={})
    plt.subplots.return_value = (Mock(), SimpleNamespace(flatten=lambda: axes))
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(pyplot=plt))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    monkeypatch.setitem(sys.modules, "analysis.utils", SimpleNamespace(load_results=Mock()))
    monkeypatch.setitem(sys.modules, "analysis.visualization", SimpleNamespace(AI2_COLORS={}, PLOT_PARAMS={}))
    monkeypatch.setitem(sys.modules, "rewardbench", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "rewardbench.constants", SimpleNamespace(SUBSET_NAME_TO_PAPER_READY={}))
    plotter = load_analysis_module("plot_per_subset_dist")
    results = pd.DataFrame(
        {
            "model": ["example/legacy", "example/ranking", "example/ratings"],
            "model_type": ["Generative"] * 3,
            "average": [0.5] * 3,
            "Factuality": [0.2, 0.4, 0.6],
            "Reasoning": [0.8, 0.6, 0.4],
            "score_w_ratings": [np.nan, False, True],
        }
    )
    plotter.generate_whisker_plot(results, output_path=None)
    plt.subplots.assert_called_once_with(1, 2, figsize=(18, 10))
    for axis, subset in zip(axes, ["Factuality", "Reasoning"]):
        axis.set_title.assert_called_once_with(subset)
        np.testing.assert_allclose(axis.violinplot.call_args.args[0], results[subset])
