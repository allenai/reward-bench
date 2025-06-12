<div align="center">
  <h1>RewardBench: Evaluating Reward Models</h1>
  <p> V2 (<strong>NEW!</strong>):
  <a href="https://huggingface.co/spaces/allenai/reward-bench">Leaderboard</a> 📐 |
  <a href="https://huggingface.co/datasets/allenai/reward-bench-2">Eval. Dataset</a> |
  <a href="https://huggingface.co/datasets/allenai/reward-bench-2-results">Results</a> 📊 | 
  <a href="https://huggingface.co/collections/allenai/reward-bench-2-683d2612a4b3e38a3e53bb51">Trained Models</a> 🏆 | 
  <a href="https://arxiv.org/abs/2506.01937"> Paper📝 </a>
</p>

  <p> V1:
  <a href="https://huggingface.co/spaces/allenai/reward-bench">Leaderboard</a> 📐 |
  <a href="https://huggingface.co/datasets/allenai/reward-bench">Eval. Dataset</a> |
  <a href="https://huggingface.co/datasets/allenai/preference-test-sets">Existing Test Sets</a> |
  <a href="https://huggingface.co/datasets/allenai/reward-bench-results">Results</a> 📊 |
  <a href="https://arxiv.org/abs/2403.13787"> Paper📝</a>
</p>
  <img width="1280" alt="Github RewardBench Logo" src="https://github.com/allenai/reward-bench/assets/10695622/39b213ba-9971-4338-b5f9-8e042d22d8fc" style="margin-left:'auto' margin-right:'auto' display:'block' "/>
</div>
<p align="center">
  <a href="https://github.com/allenai/reward-bench/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/reward-bench">
  </a>
  <a href="https://pypi.org/project/rewardbench/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/rewardbench">
  </a>
</p>


---

**RewardBench** is a benchmark designed to evaluate the capabilities and safety of reward models (including those trained with Direct Preference Optimization, DPO).
The repository includes the following:
* Common inference code for a variety of reward models (Starling, PairRM, OpenAssistant, DPO, and more).
* Common dataset formatting and tests for fair reward model inference.
* Analysis and visualization tools.

The three primary scripts to generate results (more in `scripts/`):
1. `scripts/run_rm.py`: Run evaluations for reward models.
2. `scripts/run_dpo.py`: Run evaluations for direct preference optimization (DPO) models (and other models using implicit rewards, such as KTO).
3. `scripts/run_v2.py`: Run evaluations for RewardBench 2, with special data handling for best-of-4 and Ties data.

## Quick Usage
RewardBench lets you quickly evaluate any reward model on any preference set. 
It also will detect if a instruction dataset is passed (by checking for not having `chosen`/`rejected`, and having `messages`) -- for these, just a model outputs are logged (not accuracy).

To install for quick usage, install with pip as:
```
pip install rewardbench
```
**To run RewardBench 2, you can run the following command, substituting the model you would like to run and adding any additional model-specific parameters, which can be found in the [eval configs](https://github.com/allenai/reward-bench/blob/main/scripts/configs/eval_configs.yaml) in `scripts/configs/eval_configs.yaml`**
```
python scripts/run_v2.py --model={yourmodel}
```

Generative models can be run on RewardBench 2 either with a rankings-based prompt (comparing 4 responses in one go, the default) or a ratings-based prompt (scoring each response separately then recombining, run with `--score_w_ratings` flag). Note that our Ties subset, new in RewardBench 2, has up to 20+ completions to score per-prompt, so the code enforces that it runs in the ratings setting. For more information, see `scripts/run_generative_v2.py`. To add a custom prompt for your model, feel free to open a PR.
```
python scripts/run_generative_v2.py --model={yourmodel}
```

Or, to run RewardBench instead, run the following:
```
rewardbench --model={yourmodel} --dataset={yourdataset} --batch_size=8
```
For a DPO model, pass --ref_model={} and the script will automatically route there.
Automatically uses Tokenizers chat templates, but can also use fastchat conv templates.

To run the core Reward Bench evaluation set, run:
```
rewardbench --model={yourmodel}
```

Examples:
1. Normal operation
```
rewardbench --model=OpenAssistant/reward-model-deberta-v3-large-v2 --dataset=allenai/ultrafeedback_binarized_cleaned --split=test_gen --chat_template=raw
```
2. DPO model from local dataset (note `--load_json`)
```
rewardbench --model=Qwen/Qwen1.5-0.5B-Chat --ref_model=Qwen/Qwen1.5-0.5B --dataset=/net/nfs.cirrascale/allennlp/jacobm/herm/data/berkeley-nectar-binarized-preferences-random-rejected.jsonl --load_json
```

*Experimental*: Generative RMs can be run from the pip install by running:
```
pip install rewardbench[generative]
```
And then:
```
rewardbench-gen --model={}
```
For more information, see `scripts/run_generative.py`. 
The extra requirement for local models is VLLM and the requesite API for API models (OpenAI, Anthropic, and Together are supported).

### Logging

The CLI comes with multiple advanced saving features for **model outputs** and **accuracy scores**. 
These can be tied in metadata to reward models you own or uploaded as separate datasets to HuggingFace, such as for rejection sampling.
For example, the following command does both:
```
rewardbench --model vwxyzjn/reward_modeling__EleutherAI_pythia-14m --batch_size 128 --tokenizer=EleutherAI/pythia-14m --push_results_to_hub --upload_model_metadata_to_hf --chat_template raw
```
Or, for an instruction dataset:
```
rewardbench --model vwxyzjn/reward_modeling__EleutherAI_pythia-14m --dataset HuggingFaceH4/no_robots --split test --batch_size 128 --tokenizer=EleutherAI/pythia-14m --push_results_to_hub --chat_template raw
```
(Note that chat templates only need to be specififed for older models)

The key commands are:
* `--push_results_to_hub` which uploads a dataset of scores and correctness.
* ` --upload_model_metadata_to_hf` adds results directly to model.

For an example of a model with accuracy metadata, look [here](https://huggingface.co/vwxyzjn/rm_zephyr_new).
For an example of the outputs from a preference dataset, look [here](https://huggingface.co/datasets/natolambert/rewardbench_eval_2339270924_2339270924), and for instructions, look [here](https://huggingface.co/datasets/natolambert/rewardbench_eval_0329290924).

This currently only works with DPO models for preference datasets, such as:
```
rewardbench --model Qwen/Qwen1.5-0.5B-Chat --ref_model Qwen/Qwen1.5-0.5B  --batch_size 128 --tokenizer=EleutherAI/pythia-14m --push_results_to_hub --upload_model_metadata_to_hf --chat_template raw
```
Open an issue if you would like complete functionality.

## Full Installation
To install from source, please install `torch` on your system, and then install the following requirements.
```
pip install -e .
```
Optinally, for generative scripts, run:
```
pip install -e ".[generative]"
```
Add the following to your `.bashrc`:
```
export HF_TOKEN="{your_token}"
```

## Training

For training, we recommend using [`open-instruct`](https://github.com/allenai/open-instruct).

## Contribute Your Model

For now, in order to contribute your model to the leaderboard, open an issue with the model name on HuggingFace (you can still evaluate local models with RewardBench, see below).
If custom code is needed, please open a PR that enables it in our inference stack (see [`rewardbench/models`](https://github.com/allenai/reward-bench/tree/main/rewardbench/models) for more information).

# Evaluating Models

For reference configs, see `scripts/configs/eval_configs.yaml`.
For reference on Chat Templates, many models follow the base / sft model terminology [here](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py).
A small model for debugging is available at `natolambert/gpt2-dummy-rm`.

The core scripts automatically evaluate our core evaluation set. To run these on [existing preference sets](https://huggingface.co/datasets/allenai/pref-test-sets), add the argument `--pref_sets`.

## Running Reward Models

To run individual models with `scripts/run_rm.py`, use any of the following examples:
```
python scripts/run_rm.py --model=openbmb/UltraRM-13b --chat_template=openbmb --batch_size=8
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
Note: for AI2 users, you must set `beaker secret write HF_TOKEN <your_write_token_here>` to make the scripts work.

Models using the default abstraction `AutoModelForSequenceClassification.from_pretrained` can also be loaded locally. Expanding this functionality is TODO. E.g.
```
python scripts/run_rm.py --model=/net/nfs.cirrascale/allennlp/hamishi/EasyLM/rm_13b_3ep --chat_template=tulu --batch_size=8
```

## Running DPO Models

And for DPO:
```
python scripts/run_dpo.py --model=stabilityai/stablelm-zephyr-3b --ref_model=stabilityai/stablelm-3b-4e1t --batch_size=8
python scripts/run_dpo.py --model=stabilityai/stablelm-2-zephyr-1_6b --ref_model=stabilityai/stablelm-2-1_6b --batch_size=16
```

## Ensembling RMs
For reward models already in RewardBench, you can run an offline ensemble test to approximate using multiple reward models in your system. To try this, you can run:
```
python analysis/run_ensemble_offline.py --models sfairXC/FsfairX-LLaMA3-RM-v0.1 openbmb/Eurus-RM-7b Nexusflow/Starling-RM-34B
```

## Running Generative RMs (LLM-as-a-judge)
Local and API models are supported. For example, run OpenAI's models like:
```
python scripts/run_generative.py --model=gpt-3.5-turbo-0125
```
Local models are loaded from HuggingFace, though some are also available via Together's API. Run Llama 3 locally with
```
python scripts/run_generative.py --model=meta-llama/Llama-3-70b-chat-hf --force_local
```
Or, with Together's API with:
```
python scripts/run_generative.py --model=meta-llama/Llama-3-70b-chat-hf
```

We are adding support for generative ensembles (only via API for now), run with:
```
python scripts/run_generative.py --model gpt-3.5-turbo-0125 claude-3-sonnet-20240229 meta-llama/Llama-3-70b-chat-hf
```
Note: these must be an odd number of models > 1.

## Creating Best of N (BoN) rankings

To create the ranking across the dataset, run (best_of 8 being placeholder, 16 should be fine as eval logic will handle lower best of N numbers):
```
python scripts/run_bon.py --model=OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 --chat_template=oasst_pythia --best_of=8 --debug
```
## Getting Leaderboard Section Scores

**Important**: We use prompt-weighed scores for the sections Chat, Chat Hard, Safety, and Reasoning (with math equalized to code here) to avoid assigning too much credit to small subsets (e.g. MT Bench ones). Use the following code to compute the scores for each category, assuming `RewardBench` is installed:
```
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

metrics = {
  "alpacaeval-easy": 0.5,
  "alpacaeval-hard": 0.7052631578947368,
  "alpacaeval-length": 0.5894736842105263,
  "chat_template": "tokenizer",
  "donotanswer": 0.8235294117647058,
  "hep-cpp": 0.6280487804878049,
  "hep-go": 0.6341463414634146,
  "hep-java": 0.7073170731707317,
  "hep-js": 0.6646341463414634,
  "hep-python": 0.5487804878048781,
  "hep-rust": 0.6463414634146342,
  "llmbar-adver-GPTInst": 0.391304347826087,
  "llmbar-adver-GPTOut": 0.46808510638297873,
  "llmbar-adver-manual": 0.3695652173913043,
  "llmbar-adver-neighbor": 0.43283582089552236,
  "llmbar-natural": 0.52,
  "math-prm": 0.2953020134228188,
  "model": "PKU-Alignment/beaver-7b-v1.0-cost",
  "model_type": "Seq. Classifier",
  "mt-bench-easy": 0.5714285714285714,
  "mt-bench-hard": 0.5405405405405406,
  "mt-bench-med": 0.725,
  "refusals-dangerous": 0.97,
  "refusals-offensive": 1,
  "xstest-should-refuse": 1,
  "xstest-should-respond": 0.284
}

# Calculate and print the scores per section
scores_per_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, metrics)
print(scores_per_section)
```

## Repository structure

```
├── README.md                   <- The top-level README for researchers using this project
├── analysis/                   <- Directory of tools to analyze RewardBench results or other reward model properties
├── rewardbench/                <- Core utils and modeling files
|   ├── models/                     ├── Standalone files for running existing reward models
|   └── *.py                        └── RewardBench tools and utilities
├── scripts/                    <- Scripts and configs to evaluate reward models
├── tests                       <- Unit tests
├── Dockerfile                  <- Build file for reproducible and scaleable research at AI2
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
└── setup.py                    <- Makes project pip installable (pip install -e .) so `alignment` can be imported
```

## Maintenance

This section is designed for AI2 usage, but may help others evaluating models with Docker.

### Updating the docker image 

When updating this repo, the docker image should be rebuilt to include those changes. 
For AI2 members, please update the list below with any images you use regularly.
For example, if you update `scripts/run_rm.py` and include a new package (or change a package version), you should rebuild the image and verify it still works on known models.

To update the image, run these commands in the root directory of this repo:
1. `docker build -t <local_image_name> . --platform linux/amd64`
2. `beaker image create <local_image_name> -n <beaker_image_name>`

Notes: Do not use the character - in image names for beaker,

When updating the `Dockerfile`, make sure to see the instructions at the top to update the base cuda version. 

We recently switched to automatic beaker image building workflows. 
You can use this image, or the last image with the previous Dockerfile
- `nathanl/rewardbench_auto`: Automatic image [here](https://beaker.org/im/01J60RQ6Y1KGNAD0NEPK01K03T/details).
- `nathanl/rb_v23`, Jul. 2024: Include support for bfloat16 models from command line

## Citation
Please cite our work with the following:
```
@misc{lambert2024rewardbench,
      title={RewardBench: Evaluating Reward Models for Language Modeling}, 
      author={Nathan Lambert and Valentina Pyatkin and Jacob Morrison and LJ Miranda and Bill Yuchen Lin and Khyathi Chandu and Nouha Dziri and Sachin Kumar and Tom Zick and Yejin Choi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2403.13787},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@misc{malik2025rewardbench2advancingreward,
      title={RewardBench 2: Advancing Reward Model Evaluation}, 
      author={Saumya Malik and Valentina Pyatkin and Sander Land and Jacob Morrison and Noah A. Smith and Hannaneh Hajishirzi and Nathan Lambert},
      year={2025},
      eprint={2506.01937},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01937}, 
}
```
