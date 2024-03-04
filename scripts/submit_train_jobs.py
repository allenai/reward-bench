import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/beaker_train.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

cluster = "ai2/allennlp-cirrascale"
num_gpus = 4
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "high"

with open("scripts/configs/train_configs.yaml", "r") as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)

# TODO: explain
models = [
    "allenai/tulu-2-7b",
    "meta-llama/Llama-2-7b-chat-hf",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

datasets = [
    "/net/nfs.cirrascale/allennlp/jacobm/herm/data/argilla-ultrafeedback-binarized-preferences-cleaned.jsonl",
    "/net/nfs.cirrascale/allennlp/jacobm/herm/data/berkeley-nectar-binarized-preferences-random-rejected.jsonl",
]


# ----------------------- dataset comparison -----------------------
for model in models:
    model_config = configs[model]
    d1['tasks'][0]['resources']['gpuCount'] = model_config["num_gpus"]
    for dataset in datasets:
        d = copy.deepcopy(d1)

        # name and description
        model_stem = model.replace('/', '-')
        if ".jsonl" in dataset:
            dataset_stem = dataset.split("/")[-1].replace('.jsonl', '')
        exp_name = f"herm_train-rm_{model_stem}_{dataset_stem}"
        d['description'] = exp_name
        d['tasks'][0]['name'] = exp_name

        GRADIENT_ACC_STEPS=model_config["total_batch_size"]/num_gpus/model_config["batch_size_per_gpu"]

        d["tasks"][0]["arguments"][0] = (
                f"deepspeed --include localhost:{','.join(range(num_gpus))} scripts/train_rm_trainer.py"
                " --deepspeed ds_configs/stage3_no_offloading.conf"
                f" --model_name_or_path {model}"
                f" --tokenizer {model_config['tokenizer']}"
                f" --batch_size {model_config['batch_size']}"
                f" --dataset_name {dataset}"
                f" --max_seq_length {model_config['max_seq_len']}"
                " --preprocessing_num_workers 64"
                " --do_train"
                " --use_flash_attn" if model_config['use_flash_attn'] else ""
                " --bf16" if model_config['bf16'] else ""
                f" --per_device_train_batch_size {model_config['batch_size_per_gpu']}"
                f" --gradient_accumulation_steps {GRADIENT_ACC_STEPS}"
                " --learning_rate 1e-5"
                " --lr_scheduler_type linear"
                " --warmup_ratio 0.03"
                " --weight_decay 0."
                " --evaluation_strategy no"
                " --logging_steps 1"
                " --save_strategy epoch"
                " --seed 123409876"
                " --num_train_epochs 1"
                f" --output_dir /output"
                " --use_slow_tokenizer"
                " --overwrite_output_dir &&"
                # " TODO: ADD EVAL COMMAND HERE"
            )

        fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/herm".format(fn)
        subprocess.Popen(cmd, shell=True)