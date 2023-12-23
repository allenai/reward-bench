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
import logging
import sys

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from fastchat.conversation import get_conv_template
from transformers import pipeline

from herm import prepare_dialogue

# data repo to upload results
EVAL_REPO = "ai2-rlhf-collab/rm-benchmark-results"


def push_results_to_hub(hub_path, output_path):
    api = HfApi()
    api.upload_folder(
        folder_path=output_path,
        path_in_repo=hub_path,
        repo_id=SFT_RM_REPO,
        repo_type="dataset",
    )
    logging.info(f"Uploaded results to www.huggingface.co/datasets/{hub_path}")


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="natolambert/gpt2-dmummy-rm", help="path to model")
    parser.add_argument("--chat_template", type=str, default="chatml", help="path to chat template")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    ###############
    # Setup logging
    ###############
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    ############################
    # Load dataset from ai2-rlhf-collab/rm-benchmark-dev, "filtered" split
    ############################
    logger.info("*** Load dataset ***")
    dataset = load_dataset("ai2-rlhf-collab/rm-benchmark-dev", split="filtered")
    dataset = dataset.map(
        prepare_dialogue,
        fn_kwargs={"dialogue_template": conv},
    )

    import ipdb

    ipdb.set_trace()
    ############################
    # Load reward model pipeline
    ############################
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": 4,  # eval_args.inference_batch_size,
        "truncation": True,
        "padding": True,
        "max_length": 2048,
        "function_to_apply": "none",  # Compute raw logits
        "return_token_type_ids": False,
    }

    accelerator = Accelerator()
    current_device = accelerator.process_index
    reward_pipe = pipeline(
        "text-classification",
        model=args.model,
        revision="main",
        model_kwargs={
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch.float16,
        }
        if torch.cuda.is_available()
        else None,
    )

    # def collator(data):
    #     return dict((key, [d[key] for d in data]) for key in data[0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        # collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )

    for step, batch in enumerate(tqdm(dataloader), desc="RM batch steps"):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        texts_chosen
        rewards_chosen = reward_pipe()


if __name__ == "__main__":
    main()
