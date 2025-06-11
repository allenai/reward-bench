import argparse
import copy
import os
import subprocess
import uuid
from datetime import date

import yaml

# Base Beaker configuration as a Python dictionary
BASE_CONFIG = {
    "version": "v2",
    "budget": "ai2/oe-adapt",
    "description": "Best of N ranking experiment",
    "tasks": [
        {
            "envVars": [
                {"name": "HF_TOKEN", "secret": "HF_TOKEN"},
                {"name": "CUDA_DEVICE_ORDER", "value": "PCI_BUS_ID"},
            ],
            "command": ["/bin/sh", "-c"],
            "name": "bon_ranking",
            "image": {"beaker": "your-org/bon-ranking"},
            "constraints": {
                "cluster": ["ai2/jupiter-cirrascale-2", "ai2/saturn-cirrascale", "ai2/neptune-cirrascale"]
            },
            "context": {"priority": "normal"},
            "datasets": [{"mountPath": "/weka/oe-adapt-default", "source": {"weka": "oe-adapt-default"}}],
            "resources": {"gpuCount": 1},
            "arguments": ["python scripts/run_v2.py"],
        }
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Beaker-specific arguments
    parser.add_argument("--image", type=str, default="nathanl/rewardbench_auto", help="Beaker image to use")
    parser.add_argument(
        "--cluster",
        nargs="+",
        default=["ai2/jupiter-cirrascale-2", "ai2/saturn-cirrascale", "ai2/neptune-cirrascale"],
        help="Beaker cluster to use",
    )
    parser.add_argument("--priority", type=str, default="normal", help="Priority of the job")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--workspace", type=str, default="ai2/reward-bench-v2", help="Beaker workspace")
    parser.add_argument("--mount", type=str, default="/weka/oe-adapt-default/", help="Mount")
    parser.add_argument("--source", type=str, default="oe-adapt-default", help="Source")

    # Required experiment parameters
    parser.add_argument("--dataset", type=str, default="allenai/reward-bench-2", help="Dataset to use")
    parser.add_argument("--model", type=str, required=True, help="Model to use")

    # Optional experiment parameters
    parser.add_argument("--revision", type=str, default=None, help="model revision")
    parser.add_argument("--tokenizer", type=str, help="Path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, help="Path to chat template")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--do_not_save", action="store_true", help="Do not save results to hub (for debugging)")
    # parser.add_argument("--trust_remote_code", action="store_true", help="Directly load model instead of pipeline")
    parser.add_argument("--debug", action="store_true", help="Debug on small subset of data")

    return parser.parse_args()


def create_experiment_name(args):
    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1].split(".jsonl")[0]
    today = date.today().strftime("%m%d%Y")
    unique_id = str(uuid.uuid4())[:8]
    if args.revision:
        model_name = args.revision
    return f"rb2_{dataset_name}_{model_name}_{unique_id}_{today}".replace("/", "-")[:128]


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("beaker_configs/bon_experiments", exist_ok=True)

    # Create experiment config
    config = copy.deepcopy(BASE_CONFIG)

    # Set experiment name and description
    name = create_experiment_name(args)
    config["description"] = name
    config["tasks"][0]["name"] = name

    # Configure cluster and resources
    config["tasks"][0]["image"]["beaker"] = args.image
    config["tasks"][0]["constraints"]["cluster"] = args.cluster
    config["tasks"][0]["context"]["priority"] = args.priority
    config["tasks"][0]["resources"]["gpuCount"] = args.num_gpus

    if (args.revision and "70b" in args.revision) or ("70B" in args.model) or ("72B" in args.model):
        config["tasks"][0]["resources"]["gpuCount"] = 4
        config["tasks"][0]["constraints"]["cluster"] = ["ai2/jupiter-cirrascale-2"]

    with open("scripts/configs/eval_configs.yaml", "r") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(configs)

    # get model from config keys
    models_to_evaluate = list(configs.keys())

    if args.model is not None:
        if args.model in models_to_evaluate:
            models_to_evaluate = [args.model]
        else:
            raise ValueError(f"Model {args.model} not found in configs")

    model_config = configs[args.model]

    if "torch_dtype" in model_config:
        if model_config["torch_dtype"] == "torch.bfloat16" or model_config["torch_dtype"] == "bfloat16":
            eval_bfloat16 = True
    else:
        eval_bfloat16 = False

    config["tasks"][0]["arguments"][0] = (
        f"python scripts/run_v2.py"
        f" --model {args.model}"
        f" --dataset {args.dataset}"
        f" --batch_size {model_config['batch_size']}"
    )
    if model_config["tokenizer"] is not None:
        config["tasks"][0]["arguments"][0] += f" --tokenizer {model_config['tokenizer']}"
    if model_config["chat_template"] is not None:
        config["tasks"][0]["arguments"][0] += f" --chat_template {model_config['chat_template']}"
    if model_config["trust_remote_code"]:
        config["tasks"][0]["arguments"][0] += " --trust_remote_code"
    if eval_bfloat16:
        config["tasks"][0]["arguments"][0] += " --torch_dtype=bfloat16"
    if "quantized" in model_config and model_config["quantized"]:
        config["tasks"][0]["arguments"][0] += " --quantized"

    # for run_rm only, for now, and gemma-2-27b RMs
    if "attention_implementation" in model_config:
        config["tasks"][0]["arguments"][0] += f" --attn_implementation {model_config['attention_implementation']}"

    # Optional parameters mapping
    optional_params = {
        "revision": " --revision",
        # "batch_size": " --batch_size"
    }

    # Add optional parameters if specified
    for param_name, cmd_arg in optional_params.items():
        value = getattr(args, param_name)
        if value is not None:
            if isinstance(value, str) and any(char.isspace() for char in value):
                config["tasks"][0]["arguments"][0] += f"{cmd_arg} '{value}'"
            else:
                config["tasks"][0]["arguments"][0] += f"{cmd_arg} {value}"

    # Add flags if they're True
    flag_params = ["do_not_save", "debug"]

    for flag in flag_params:
        if getattr(args, flag):
            config["tasks"][0]["arguments"][0] += f" --{flag}"

    # Join command parts
    # config["tasks"][0]["arguments"][0] = " ".join(cmd_parts)

    # Write config file
    config_path = f"beaker_configs/bon_experiments/{name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Submit to Beaker
    print(f"Submitting experiment: {name}")
    beaker_cmd = f"beaker experiment create {config_path} --workspace {args.workspace}"
    subprocess.Popen(beaker_cmd, shell=True)


if __name__ == "__main__":
    main()
