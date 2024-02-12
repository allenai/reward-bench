import copy
import subprocess
import yaml
import os
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_eval.yaml", "r") as f:
    d1 = yaml.load(f.read(), Loader=yaml.FullLoader)
with open("scripts/default_eval_configs.yaml", "r") as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print(configs)

cluster = "ai2/allennlp-cirrascale"
# cluster = "ai2/mosaic-cirrascale"
image = "jacobm/herm"
num_gpus = 1
upload_to_hub = False
eval_on_pref_sets = False
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "HF Token does not exist -- run `Export HF_TOKEN=<your_write_token_here>'"

d1["tasks"][0]["image"] = image
d1["tasks"][0]["context"]["cluster"] = cluster
d1["tasks"][0]["context"]["priority"] = "high"
d1["tasks"][0]["resources"]["gpuCount"] = num_gpus
d1["tasks"][0]["envVars"][6]["value"] = HF_TOKEN

models_to_evaluate = [
    "openbmb/UltraRM-13b",
    "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
    "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "weqweasdas/hh_rlhf_rm_open_llama_3b",
    "llm-blender/PairRM-hf",
    "berkeley-nest/Starling-RM-7B-alpha",
    "stanfordnlp/SteamSHP-flan-t5-xl",
    "PKU-Alignment/beaver-7b-v1.0-reward",
    "PKU-Alignment/beaver-7b-v1.0-cost",
    "IDEA-CCNL/Ziya-LLaMA-7B-Reward",
]

for model in models_to_evaluate:
    model_config = configs[model]
    if eval_on_pref_sets:
        experiment_group = "common-preference-sets"
    else:
        experiment_group = "herm-preference-sets"
    print(f"Submitting evaluation for model: {model_config['model']} on {experiment_group}")
    d = copy.deepcopy(d1)

    name = f'herm_eval_for_{model_config["model"]}_on_{experiment_group}'.replace("/", "-")
    d["description"] = name
    d["tasks"][0]["name"] = name

    d["tasks"][0]["arguments"][0] = (
        f"python scripts/run_rm.py"
        f" --model {model_config['model']}"
        f" --tokenizer {model_config['tokenizer']}"
        f" --chat_template {model_config['chat_template']}"
        f" --batch_size {model_config['batch_size']}"
    )

    if model_config["direct_load"]:
        d["tasks"][0]["arguments"][0] += " --direct_load"
    if model_config["trust_remote_code"]:
        d["tasks"][0]["arguments"][0] += " --trust_remote_code"
    if not upload_to_hub:
        d["tasks"][0]["arguments"][0] += " --do_not_save"
    if eval_on_pref_sets:
        d["tasks"][0]["arguments"][0] += " --pref_sets"

    fn = "beaker_configs/auto_created/{}.yaml".format(name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/herm".format(fn)
    subprocess.Popen(cmd, shell=True)
