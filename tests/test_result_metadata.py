# Copyright 2026 AllenAI. All rights reserved.
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
import ast
import io
import json
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from datasets import Dataset, load_dataset

from analysis.utils import load_results, load_scores

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_generative_v2.py"


class ResultMetadataTest(unittest.TestCase):
    def test_runner_serializes_protocol_in_both_artifacts(self):
        # Execute the actual reporting code without importing GPU/API inference dependencies.
        main = next(
            node
            for node in ast.parse(RUNNER.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        start = next(
            i
            for i, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "results_grouped" for target in node.targets)
        )
        reporting = compile(ast.Module(body=main.body[start:], type_ignores=[]), str(RUNNER), "exec")
        dataset = Dataset.from_dict(
            {"id": [10, 11, 12], "subset": ["Math", "Math", "Ties"], "results": [1.0, 0.0, 0.5]}
        )
        for ratings in (False, True):
            with self.subTest(score_w_ratings=ratings), TemporaryDirectory() as temp_dir:

                def save_artifact(data, model_name, sub_path, *args, **kwargs):
                    path = Path(temp_dir) / sub_path / f"{model_name}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
                    return str(path)

                namespace = {
                    "args": SimpleNamespace(
                        score_w_ratings=ratings,
                        chat_template=None,
                        debug=False,
                        do_not_save=True,
                        disable_beaker_save=True,
                    ),
                    "out_dataset": dataset,
                    "ties_score": 0.75,
                    "model_name": "example/judge",
                    "model_type": "Generative",
                    "np": np,
                    "logger": Mock(),
                    "save_to_hub": save_artifact,
                }
                with redirect_stdout(io.StringIO()):
                    exec(reporting, namespace)
                grouped = json.loads((Path(temp_dir) / "eval-set/example/judge.json").read_text())
                scores = json.loads((Path(temp_dir) / "eval-set-scores/example/judge.json").read_text())
                self.assertIs(grouped["score_w_ratings"], ratings)
                self.assertIs(scores["score_w_ratings"], ratings)
                self.assertEqual(grouped["Math"], 0.5)
                self.assertEqual(grouped["Ties"], 0.75)
                self.assertEqual(scores["results"], [1.0, 0.0, 0.5])
                self.assertEqual(scores["id"], [10, 11, 12])
                loaded_scores = load_scores(temp_dir, "eval-set-scores")
                self.assertEqual(loaded_scores["score_w_ratings"].tolist(), [ratings] * 3)
                self.assertEqual(loaded_scores["results"].tolist(), scores["results"])

    def test_loading_old_new_and_mixed_results_preserves_averages(self):
        for protocols in ((None,), (False, True), (None, False, True)):
            with self.subTest(protocols=protocols), TemporaryDirectory() as temp_dir:
                result_dir = Path(temp_dir) / "eval-set" / "example"
                result_dir.mkdir(parents=True)
                for index, protocol in enumerate(protocols):
                    record = {
                        "model": f"example/judge-{index}",
                        "model_type": "Generative",
                        "chat_template": None,
                        "Math": 0.2,
                        "Ties": 0.8,
                    }
                    if protocol is not None:
                        record["score_w_ratings"] = protocol
                    (result_dir / f"judge-{index}.json").write_text(json.dumps(record))
                # Keep Hugging Face's local JSON cache inside the temporary directory.
                with patch("analysis.utils.load_dataset", partial(load_dataset, cache_dir=f"{temp_dir}/cache")):
                    results = load_results(temp_dir, "eval-set").set_index("model")
                for index, protocol in enumerate(protocols):
                    row = results.loc[f"example/judge-{index}"]
                    self.assertAlmostEqual(row["average"], 0.5)
                    self.assertAlmostEqual(row["Ties"], 0.8)
                self.assertEqual(list(results.columns), ["model_type", "average", "Math", "Ties"])


if __name__ == "__main__":
    unittest.main()
