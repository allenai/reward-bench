# Holistic Evaluation of Reward Models (HERM)

This will hold scripts for generating scores and uploading results.
Two primary to generate results (more in `scripts/`):
1. `scripts/run_rm.py`: Run evaluations for reward models.
2. `scripts/run_dpo.py`: Run evaluations for direct preference optimization (DPO) models.

## Links
Dataset, space, etc coming soon.
For contributors, it can be found in this [HuggingFace org](https://huggingface.co/ai2-adapt-dev).

## Installation
Please install `torch`` on your system, and then install the following requirements.
```
pip install -e .
```
Add the following to your `.bashrc`:
```
export HF_TOKEN="{your_token}"
```

# Evaluating Models

For reference configs, see `scripts/configs/eval_configs.yaml`.
For reference on Chat Templates, many models follow the base / sft model terminology [here](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py).
A small model for debugging is available at `natolambert/gpt2-dummy-rm`.

The core scripts automatically evaluate our core evaluation set. To run these on [existing preference sets](https://huggingface.co/datasets/allenai/pref-test-sets), add the argument `--pref_sets`.

## Running Reward Models

To run individual models with `scripts/run_rm.py`, use any of the following examples:
```
python scripts/run_rm.py --model=openbmb/UltraRM-13b --chat_template=billa --batch_size=8
python scripts/run_rm.py --model=OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 --chat_template=oasst_pythia
python scripts/run_rm.py --model=PKU-Alignment/beaver-7b-v1.0-cost --chat_template=pku-align --batch_size=16
python scripts/run_rm.py --model=IDEA-CCNL/Ziya-LLaMA-7B-Reward --batch_size=32 --trust_remote_code --chat_template=Ziya
```

To run these models with AI2 infrastructure, run:
```
python scripts/submit_eval_jobs.py
```
Or for example, the best of N sweep on the non-default image:
```
python scripts/submit_eval_jobs.py --eval_on_bon --image=nathanl/herm_bon
``` 


## Running DPO Models

And for DPO:
```
python scripts/run_dpo.py --model=stabilityai/stablelm-zephyr-3b --ref_model=stabilityai/stablelm-3b-4e1t --batch_size=32
```

## Creating Best of N (BoN) rankings

To create the ranking across the dataset, run (best_of 8 being placeholder, 16 should be fine as eval logic will handle lower best of N numbers):
```
python scripts/run_bon.py --model=OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 --chat_template=oasst_pythia --best_of=8 --debug
```

## Repository structure

```
├── README.md                   <- The top-level README for researchers using this project
├── analysis/                   <- Directory of tools to analyze HERM results or other reward model properties
├── herm/                       <- Core utils and modeling files
|   ├── models/                     ├── Standalone files for running existing reward models
|   └── *.py                        └── HERM tools and utilities
├── scripts/                    <- Scripts and configs to train and evaluate reward models
├── tests                       <- Unit tests
├── Dockerfile                  <- Build file for reproducible and scaleable research at AI2
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
└── setup.py                    <- Makes project pip installable (pip install -e .) so `alignment` can be imported
```

## Maitenence

### Updating the docker image (consider removing this section when we publicly release HERM)
When updating this repo, the docker image should be rebuilt to include those changes. 
For AI2 members, please update the list below with any images you use regularly.
For example, if you update `scripts/run_rm.py` and include a new package (or change a package version), you should rebuild the image and verify it still works on known models.

To update the image, run these commands in the root directory of this repo:
1. `docker build -t <local_image_name> . --platform linux/amd64`
2. `beaker image create <local_image_name> -n <beaker_image_name>`

Notes: Do not use the character - in image names for beaker,

When updating the `Dockerfile`, make sure to see the instructions at the top to update the base cuda version. 

In development, we have the following docker images (most recent first as it's likely what you need).
TODO: we should log the git commit affiliated with each of these, or delete them when outdated.
- `nathanl/herm_dpo`: for adding functionality with DPO sweeps, fix minor bugs (last updated 24 Feb.)
- `nathanl/herm`: for running everything, including new Starling 34B RM (last updated 22 Feb.)
- `nathanl/herm_bon_v2`: for running `run_bon.py` sweeps (last updated 21 Feb.) 
- `jacobm/herm`: for running `run_rm.py` sweeps (last updated 14 Feb.)
