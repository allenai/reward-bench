from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union


class PairPMPipeline:

    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path,).cuda() #, attn_implementation="flash_attention_2",  torch_dtype=torch.bfloat16
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer_data_format = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.tokenizer_data_format.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

        self.prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
        token_id_A = self.tokenizer.encode("A", add_special_tokens=False)
        token_id_B = self.tokenizer.encode("B", add_special_tokens=False)
        assert len(token_id_A) == 1 and len(token_id_B) == 1
        self.token_id_A = token_id_A[0]
        self.token_id_B = token_id_B[0]
        self.temperature = 1.0

    def __call__(self, prompts: List[str], candidates_A: List[str], candidates_B: List[str]):
        '''
        Input:
            prompts: [prompt1, prompt2, ..., promptn]
            candidates_A: [responseA1, responses A2, ..., responseAn]
            candidates_B: [responseB1, responses B2, ..., responseBn]
        Output:
            probs_choose_A: [P(responseA1 > responseB1 | prompt1), ...., P(responseAn > responseBn | promptn)]
        '''
        assert len(prompts) == len(candidates_A)
        assert len(candidates_A) == len(candidates_B)
        probs_choose_A = []
        for i in range(len(prompts)):
            instruction = [{"role": "user", "content": prompts[i]}]
            context = self.tokenizer_data_format.apply_chat_template(instruction, tokenize=False)
            responses = [candidates_A[i], candidates_B[i]]
        
            probs_chosen = []
    
            for chosen_position in [0, 1]:
                # we swap order to mitigate position bias
                response_A = responses[chosen_position]
                response_B = responses[1 - chosen_position]
                prompt = self.prompt_template.format(context=context, response_A=response_A, response_B=response_B)
                message = [
                    {"role": "user", "content": prompt},
                ]

                input_ids = self.tokenizer.encode(self.tokenizer.apply_chat_template(message, tokenize=False).replace(self.tokenizer.bos_token, ""), return_tensors='pt', add_special_tokens=False).cuda() 
            
                with torch.no_grad():
                    output = self.model(input_ids)
                logit_A = output.logits[0, -1, self.token_id_A].item()
                logit_B = output.logits[0, -1, self.token_id_B].item()
                # take softmax to get the probability; using numpy
                Z = np.exp(logit_A / self.temperature) + np.exp(logit_B / self.temperature)
                logit_chosen = [logit_A, logit_B][chosen_position]
                prob_chosen = np.exp(logit_chosen / self.temperature) / Z
                probs_chosen.append(prob_chosen)
            probs_choose_A.append(np.mean(probs_chosen))
        # probs_chose_B = 1 - probs_choose_A
        return probs_choose_A


'''
#### An example to use the pair-preference model.
tqdm.pandas()
ds_dir = "allenai/reward-bench"
ds = load_dataset(ds_dir, split='filtered', keep_in_memory=True)
df = pd.DataFrame(columns=['id', 'subset', 'correct'])
pair_pm = PairPMPipeline("RLHFlow/pair-preference-model-LLaMA3-8B")

for i, example in enumerate(tqdm(ds)):
    prompt = example['prompt']
    response_chosen = example["chosen"]
    response_rejected = example["rejected"]
    
    avg_prob_chosen = pair_pm([prompt], [response_chosen], [response_rejected])
    correct = 0.5 if avg_prob_chosen[0] == 0.5 else float(avg_prob_chosen[0] > 0.5)

    row = {'id': example['id'], 'subset': example['subset']}
    row['correct'] = correct
    df = df._append(row, ignore_index=True)

categories = {
    "chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
    "chat-hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                  'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
    "safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond',
               'donotanswer'],
    "reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
}

df_acc = pd.DataFrame(columns=['category', 'subset', 'accuracy'])
for category, subsets in categories.items():
    for subset in subsets:
        df_subset = df[df['subset'] == subset]
        accs = []
        acc = df_subset['correct'].values.mean()
        accs.append(acc)
        row = {'category': category, 'subset': subset, 'n': len(df_subset), 'accuracy': accs}
        df_acc = pd.concat([df_acc, pd.DataFrame(row)], ignore_index=True)
print(df_acc)

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 250,
    "xstest-should-respond": 154,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = round(100 * total_weighted_score / total_examples, 2)
        else:
            section_scores[section] = 0
    return section_scores


all_subsets = df['subset'].unique()
df_final = pd.DataFrame(columns=['attribute', 'Chat', 'Chat Hard', 'Safety', 'Reasoning'])

attribute = 'correct'
metrics = {}
for subset in all_subsets:
    df_subset = df_acc.loc[df_acc['subset'] == subset]
    acc = df_subset['accuracy'].values[0]
    metrics[subset] = acc

# Calculate and print the scores per section
scores_per_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, metrics)
row = {'attribute': attribute, **scores_per_section}
df_final = df_final._append(row, ignore_index=True)

for col in ['Chat', 'Chat Hard', 'Safety', 'Reasoning']:
    print(f"{col}: {df_final[col].values[0]}")
'''