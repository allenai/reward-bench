# Best-of-N Evaluation with RewardBench 

This folder serves for evaluating reward models' abilities in selecting the best output from a set of N outputs (i.e., best-of-N sampling, aka. rejection sampling).  


## Download BoN data 

```bash
cd bon_eval
python download_bon_data.py
```

## AlpacaEval-BoN

- LLM: Tulu-2-dpo-13b
- N: 16
- Judge: GPT-4-turbo (same as the standard AlpacaEval 2 with length control)
- Reference: GPT-3.5-turbo (`bon_data/gpt-3.5-turbo-0613.ae.json`)

We sample 16 outputs from the Tulu-2-dpo-13b LLM. The model outputs are divided into 16 files: `bon_data/alpaca_eval_n=16/virtual/tulu-2-dpo-13b.[x].json` where `[x]` is from 0 to 15. 

We use AlpacaEval to evaluate each output with the GPT-4-turbo judge. 
The reference for computing win-rates is the GPT-3.5-turbo model.  
The AlpacaEval annotations are stored in `bon_data/alpaca_eval_n=16/virtual/annotations_ref=GPT35t/tulu-2-dpo-13b.[x]/weighted_alpaca_eval_gpt4_turbo/annotations.json`. 

If you'd like to reproduce the evaluation annotations:"
```bash
for i in {0..15}
do 
  output_dir="bon_data/alpaca_eval_n=16/virtual/annotations_ref=GPT35t/tulu-2-dpo-13b.$i/"
  mkdir -p $output_dir
  alpaca_eval --reference_outputs "bon_data/gpt-3.5-turbo-0613.ae.json" \
              --model_outputs "bon_data/alpaca_eval_n=16/virtual/tulu-2-dpo-13b.$i.json" \
              --output_path $output_dir
done 
```

## Evaluation

<!-- export HF_ENDPOINT=https://hf-mirror.com -->

Run `python rm_bon_eval.py` and you will get a json named `bon_eval_results.json` in the current directory, which will be also uploaded to HuggingFace hub.

