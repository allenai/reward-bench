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
