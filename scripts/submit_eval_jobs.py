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

import argparse
import copy
import os
import subprocess
from datetime import date

import yaml

# Create argparse, for store_true variables of eval_on_pref_sets and eval_on_bon
# String image for Beaker image
# Bool default true for upload_to_hub
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--eval_on_pref_sets", action="store_true", default=False, help="Evaluate on preference sets rather than core set"
)
argparser.add_argument("--eval_dpo", action="store_true", default=False, help="Evaluate DPO model suite")
argparser.add_argument("--eval_on_bon", action="store_true", default=False, help="Evaluate on BON preference sets")
argparser.add_argument("--image", type=str, default="nathanl/herm_v6", help="Beaker image to use")
argparser.add_argument("--cluster", type=str, default="ai2/allennlp-cirrascale", help="Beaker cluster to use")
argparser.add_argument("--upload_to_hub", action="store_false", default=True, help="Upload to results to HF hub")
argparser.add_argument("--model", type=str, default=None, help="Specific model to evaluate if not sweep")
args = argparser.parse_args()

today = date.today().strftime("%m%d%Y")

with open("scripts/configs/beaker_eval.yaml", "r") as f:
    d1 = yaml.load(f.read(), Loader=yaml.FullLoader)

cluster = args.cluster
# cluster = "ai2/mosaic-cirrascale"
image = args.image
num_gpus = 1
upload_to_hub = args.upload_to_hub
eval_on_pref_sets = args.eval_on_pref_sets
eval_on_bon = args.eval_on_bon
eval_dpo = args.eval_dpo

if eval_on_bon:
    with open("scripts/configs/eval_bon_configs.yaml", "r") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
elif eval_dpo:
    with open("scripts/configs/eval_dpo_configs.yaml", "r") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
else:
    with open("scripts/configs/eval_configs.yaml", "r") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print(configs)


# assert only one of eval_on_pref_sets and eval_on_bon is True
assert not (eval_on_pref_sets and eval_on_bon), "Only one of eval_on_pref_sets and eval_on_bon can be True"
assert not (eval_on_bon and eval_dpo), "Only one of BoN and DPO can be True (separate scripts for now not implemented)"

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF Token does not exist -- run `Export HF_TOKEN=<your_write_token_here>'"

d1["tasks"][0]["image"]["beaker"] = image
d1["tasks"][0]["context"]["cluster"] = cluster
d1["tasks"][0]["context"]["priority"] = "high"
d1["tasks"][0]["resources"]["gpuCount"] = num_gpus
d1["tasks"][0]["envVars"][6]["value"] = HF_TOKEN

# get model from config keys
models_to_evaluate = list(configs.keys())

if args.model is not None:
    if args.model in models_to_evaluate:
        models_to_evaluate = [args.model]
    else:
        raise ValueError(f"Model {args.model} not found in configs")

for model in models_to_evaluate:
    model_config = configs[model]
    if eval_on_pref_sets:
        experiment_group = "common-preference-sets"
        script = "run_dpo.py" if eval_dpo else "run_rm.py"
    elif eval_on_bon:
        experiment_group = "bon-preference-sets"
        script = "run_bon.py"
    elif eval_dpo:
        experiment_group = "dpo-eval"
        script = "run_dpo.py"
    else:
        experiment_group = "herm-preference-sets"
        script = "run_rm.py"
    print(f"Submitting evaluation for model: {model} on {experiment_group}")
    d = copy.deepcopy(d1)

    name = f"herm_eval_for_{model}_on_{experiment_group}".replace("/", "-")
    d["description"] = name
    d["tasks"][0]["name"] = name

    if "num_gpus" in model_config:
        d["tasks"][0]["resources"]["gpuCount"] = model_config["num_gpus"]

    d["tasks"][0]["arguments"][0] = (
        f"python scripts/{script}"
        f" --model {model}"
        f" --tokenizer {model_config['tokenizer']}"
        f" --batch_size {model_config['batch_size']}"
    )
    if model_config["chat_template"] is not None:
        d["tasks"][0]["arguments"][0] += f" --chat_template {model_config['chat_template']}"
    if model_config["trust_remote_code"]:
        d["tasks"][0]["arguments"][0] += " --trust_remote_code"
    if not upload_to_hub:
        d["tasks"][0]["arguments"][0] += " --do_not_save"
    if eval_on_pref_sets:
        d["tasks"][0]["arguments"][0] += " --pref_sets"
    if "ref_model" in model_config:
        d["tasks"][0]["arguments"][0] += f" --ref_model {model_config['ref_model']}"
        # TODO add logic for ref free
    # use os to check if beaker_configs/auto_created exists
    if not os.path.exists("beaker_configs/auto_created"):
        os.makedirs("beaker_configs/auto_created")

    fn = "beaker_configs/auto_created/{}.yaml".format(name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/herm".format(fn)
    subprocess.Popen(cmd, shell=True)
