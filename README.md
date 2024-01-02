# Holistic Evaluation of Reward Models (HERM)

This will hold scripts for generating scores and uploading results.
Two primary scripts need to be created:
1. `run_c_rm.py`:
2. `run_dpo_rm.py`:

## Links
Dataset, space, etc coming soon.
For contributors, it can be found in this [HuggingFace org](https://huggingface.co/ai2-rlhf-collab).

## Installation
Please install `torch`` on your system, and then install the following requirements.
```
pip install -e .
```

### Older instructions
```
pip install requirements.txt
```

If issues, run the following:
Install `fastchat` partially (for `conversation.py`):
```
pip3 install "fschat[model_worker,webui]"
pip install huggingface_hub datasets
```


### Models with chat templates
For reference on Chat Templates, many models follow the base / sft model terminology [here](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py):
I was debugging with default gpt2, but the random head may be causing numerical stability issues.
Next:
```
python scripts/run_rm.py --model=openbmb/UltraRM-13b --chat_template=billa
python scripts/run_rm.py --model=OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 --chat_template=oasst_pythia --direct_load
python scripts/run_rm.py --model=OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1 --chat_template=oasst_pythia --direct_load
python scripts/run_rm.py --model=OpenAssistant/reward-model-deberta-v3-large-v2 --chat_template=raw
python scripts/run_rm.py --model=weqweasdas/hh_rlhf_rm_open_llama_3b --chat_template=Robin
python scripts/run_rm.py --model=llm-blender/PairRM --chat_template=raw #TODO needs custom code
python scripts/run_rm.py --model=berkeley-nest/Starling-RM-7B-alpha --tokenizer=meta-llama/Llama-2-7b-chat-hf --chat_template=llama-2 --direct_load --batch_size=16
```