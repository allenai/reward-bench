import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_finetune.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/general-cirrascale"
# cluster = "ai2/yizhongw-a100-80gb"
cluster = "ai2/allennlp-cirrascale"
num_gpus = 4
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "high"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

# modify here for different set of experiments
experiment_group = "dataset_comparison"
wandb_project = "open_instruct"
wandb_api_key = "Your Wandb API Key"

models = [
    "allenai/tulu-2-7b",
]

datasets = [
    "/net/nfs.cirrascale/allennlp/jacobm/herm/data/argilla-ultrafeedback-binarized-preferences-cleaned.jsonl",
    "/net/nfs.cirrascale/allennlp/jacobm/herm/data/berkeley-nectar-binarized-preferences-random-rejected.jsonl",
]


# ----------------------- dataset comparison -----------------------
for model in models:
    for dataset in datasets:
        d = copy.deepcopy(d1)

        # name and description
        model_stem = model.replace('/', '-')
        if ".jsonl" in dataset:
            dataset_stem = dataset.split("/")[-1].replace('.jsonl', '')
        exp_name = f"herm_train-rm_{model_stem}_{dataset_stem}"
        d['description'] = exp_name
        d['tasks'][0]['name'] = exp_name

        GRADIENT_ACC_STEPS=model_config["total_batch_size"]/num_gpus/model_config["batch_size_per_gpu"]))

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
                f" --output_dir {OUTPUT_DIR}"
                " --use_slow_tokenizer"
                " --overwrite_output_dir"
            )

        # model specific
        for mount_dataset in d['tasks'][0]['datasets']:
            if mount_dataset["mountPath"] == "/hf_llama_models":
                mount_dataset["source"]["beaker"] = f"Yizhongw03/hf_llama_model_{model_size}"
        if model_size == "7B":
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--per_device_train_batch_size 2", 
                "--per_device_train_batch_size 2"
            )
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--gradient_accumulation_steps 16",
                f"--gradient_accumulation_steps {128 // 2 // num_gpus}"
            )
        elif model_size == "13B":
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--per_device_train_batch_size 2", 
                "--per_device_train_batch_size 2"
            )
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--gradient_accumulation_steps 16",
                f"--gradient_accumulation_steps {128 // 2 // num_gpus}"
            )
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf",
                "--deepspeed_config_file ds_configs/stage3_offloading_accelerate.conf",
            )
        else:
            raise NotImplementedError

        fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)