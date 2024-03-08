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
import subprocess
from datetime import date

import yaml

argparser = argparse.ArgumentParser()
argparser.add_argument("--image", type=str, default="jacobm/rb_train", help="Beaker image to use")
argparser.add_argument("--cluster", type=str, default="ai2/allennlp-cirrascale", help="Beaker cluster to use")
argparser.add_argument("--model", type=str, default=None, help="Specific model to train on top of")
argparser.add_argument("--dataset", type=str, default=None, help="Specific dataset file path for training")
argparser.add_argument("--lr", type=str, default="1e-5", help="Learning rate for training")
argparser.add_argument("--num_epochs", type=str, default="1", help="Number of training epochs")
argparser.add_argument("--seed", type=int, default=123409876, help="Seed for training")
args = argparser.parse_args()


today = date.today().strftime("%m%d%Y")

with open("scripts/configs/beaker_train.yaml", "r") as f:
    default_yaml = f.read()
d = yaml.load(default_yaml, Loader=yaml.FullLoader)

with open("scripts/configs/train_configs.yaml", "r") as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
model_config = configs[args.model]

# name and description
model_stem = args.model.replace("/", "-")
if ".jsonl" in args.dataset:
    dataset_stem = args.dataset.split("/")[-1].replace(".jsonl", "")
else:
    dataset_stem = args.dataset
exp_name = f"herm_train-rm_{model_stem}_{dataset_stem}"

d["description"] = exp_name
d["tasks"][0]["context"]["cluster"] = args.cluster
d["tasks"][0]["context"]["priority"] = "high"
d["tasks"][0]["name"] = exp_name
d["tasks"][0]["image"]["beaker"] = args.image
d["tasks"][0]["resources"]["gpuCount"] = model_config["num_gpus"]

GRADIENT_ACC_STEPS = int(
    model_config["total_batch_size"] / model_config["num_gpus"] / model_config["batch_size_per_gpu"]
)

optional_configs = ""
if model_config["bf16"]:
    optional_configs += " --bf16"
if model_config["use_flash_attn"]:
    optional_configs += " --use_flash_attn"

d["tasks"][0]["arguments"][0] = (
    f"deepspeed --include localhost:{','.join(str(n) for n in range(model_config['num_gpus']))} "
    " scripts/train_rm_trainer.py"
    " --deepspeed scripts/configs/stage3_no_offloading.conf"
    f" --model_name_or_path {args.model}"
    f" --tokenizer {model_config['tokenizer']}"
    f" --dataset_name {args.dataset}"
    f" --max_seq_length {model_config['max_seq_len']}"
    " --preprocessing_num_workers 64"
    f" --do_train {optional_configs}"
    f" --per_device_train_batch_size {model_config['batch_size_per_gpu']}"
    f" --gradient_accumulation_steps {GRADIENT_ACC_STEPS}"
    f" --learning_rate {args.lr}"
    " --lr_scheduler_type linear"
    " --warmup_ratio 0.03"
    " --weight_decay 0."
    " --evaluation_strategy no"
    " --logging_steps 1"
    " --save_strategy epoch"
    f" --seed {args.seed}"
    f" --num_train_epochs {args.num_epochs}"
    f" --output_dir /output"
    " --use_slow_tokenizer"
    " --overwrite_output_dir"
    " --output_dir /output"
)

fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
file = open(fn, "w")
yaml.dump(d, file, default_flow_style=True)
file.close()

cmd = "beaker experiment create {} --workspace ai2/herm".format(fn)
subprocess.Popen(cmd, shell=True)
