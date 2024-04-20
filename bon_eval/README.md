# Best-of-N Evaluation with RewardBench 

This folder serves for evaluating reward models' abilities in selecting the best output from a set of N outputs (i.e., best-of-N sampling, aka. rejection sampling).  

## AlpacaEval-BoN

- LLM: Tulu-2-dpo-13b
- N: 16
- Judge: GPT-4-turbo (same as the standard AlpacaEval 2 with length control)
- Reference: GPT-3.5-turbo (`gpt-3.5-turbo-0613.ae.json`)

We sample 16 outputs from the Tulu-2-dpo-13b LLM. The model outputs are divided into 16 files: `alpaca_eval_n=16/annotations/tulu-2-dpo-13b.[x].json` where `[x]` is from 0 to 15. 

We use AlpacaEval to evaluate each output with the GPT-4-turbo judge. 
The reference for computing win-rates is the GPT-3.5-turbo model.  
The AlpacaEval annotations are stored in `alpaca_eval_n=16/annotations/annotations_ref=GPT35t/tulu-2-dpo-13b.[x]/weighted_alpaca_eval_gpt4_turbo/annotations.json`. 

If you'd like to reproduce the evaluation annotations:"
```bash
for i in {0..15}
do 
  output_dir="alpaca_eval_n=16/annotations/annotations_ref=GPT35t/tulu-2-dpo-13b.$i/"
  mkdir -p $output_dir
  alpaca_eval --reference_outputs "gpt-3.5-turbo-0613.ae.json" \
              --model_outputs "alpaca_eval_n=16/annotations/tulu-2-dpo-13b.$i.json" \
              --output_path $output_dir
done 
```

## Evaluation

<!-- export HF_ENDPOINT=https://hf-mirror.com -->

```bash
python rm_bon_eval.py openbmb/UltraRM-13b
```