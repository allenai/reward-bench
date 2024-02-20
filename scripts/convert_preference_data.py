import json
from statistics import mean
from random import Random
from datasets import load_dataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_dataset', type=str)
parser.add_argument('--split', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--seed', type=int, default=42)
# for now, max 5 million samples.
parser.add_argument('--max_samples', type=int, default=5_000_000)
args = parser.parse_args()

dataset = load_dataset(args.input_dataset, split=args.split)
dataset = dataset.shuffle(args.seed).select(range(min(args.max_samples, len(dataset))))
new_data = []
random_gen = Random(args.seed)

if args.input_dataset == 'nvidia/HelpSteer':
    # group by prompt
    prompts = {}
    for sample in dataset:
        prompt = sample['prompt']
        if prompt not in prompts:
            prompts[prompt] = []
        prompts[prompt].append(sample)
    # filter out prompts with less than 2 responses
    prompts = {k: v for k, v in prompts.items() if len(v) > 1}

    for prompt, samples in prompts.items():
        samples = sorted(samples, key=lambda x: mean([
            x['helpfulness'],
            x['correctness'],
            x['coherence'],
            x['complexity'],
            # x['verbosity']  - we don't really care about verbosity
        ]))
        chosen = samples[0]
        rejected = random_gen.choice(samples[1:])
        chosen =  [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': chosen['response']},
        ]
        rejected =  [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': rejected['response']},
        ]
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'helpsteer'
        })
elif args.input_dataset == 'berkeley-nest/Nectar':
    for sample in dataset:
        prompt = sample['prompt'].replace("Human: ", "").replace("Assistant: ", "").strip()
        answers = sorted(sample['answers'], key=lambda x: x['rank'])
        # chosen = answers[0]['answer']
        # rejected = random_gen.choice(answers[1:])['answer']
        pairs = [
            (0,5),
            (1,5),
            (0,6),
            (1,6)
        ]
        # pairs = [
        #     (0,1),
        #     (0,2),
        #     (0,3),
        #     (0,4),
        #     (0,5),
        #     (0,6),
        # ]
        # for (i, j) in pairs:
        for i in range(len(answers) - 1):
            for j in range(i + 1, len(answers)):
                chosen = answers[i]['answer']
                rejected = answers[j]['answer']
                chosen =  [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': chosen},
                ]
                rejected =  [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': rejected},
                ]
                new_data.append({
                    'chosen': chosen,
                    'rejected': rejected,
                    'source': 'nectar'
                })
elif args.input_dataset == 'argilla/ultrafeedback-binarized-preferences-cleaned':
    for sample in dataset:
        chosen = sample['chosen']
        rejected = sample['rejected']
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'argilla-ultrafeedback'
        })
elif args.input_dataset == 'HuggingFaceH4/ultrafeedback_binarized':
    argilla_dataset = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', split='train')
    prompts_in_argilla = set([x['prompt'] for x in argilla_dataset])
    for sample in dataset:
        prompt = sample['prompt']
        if prompt in prompts_in_argilla:
            chosen = sample['chosen']
            rejected = sample['rejected']
            new_data.append({
                'chosen': chosen,
                'rejected': rejected,
                'source': 'h4-ultrafeedback'
            })
elif args.input_dataset == 'stanfordnlp/SHP' or args.input_dataset == 'stanfordnlp/SHP-2':
    for el in dataset:
        prompt = {'content': el['history'], 'role': 'user'}
        label = el['labels']
        if label == 1:
            chosen = {'content': el['human_ref_A'], 'role': 'assistant'}
            rejected = {'content': el['human_ref_B'], 'role': 'assistant'}
        else:
            chosen = {'content': el['human_ref_B'], 'role': 'assistant'}
            rejected = {'content': el['human_ref_A'], 'role': 'assistant'}
        data = {}
        data = {'chosen': [prompt, chosen], 'rejected': [prompt, rejected]}
        data['source'] = 'shp'
        new_data.append(data)
elif args.input_dataset == 'Intel/orca_dpo_pairs':
    for sample in dataset:
        prompt = {'role': 'user', 'content': sample['question']}
        chosen = [prompt, {'role': 'assistant', 'content': sample['chosen']}]
        rejected = [prompt, {'role': 'assistant', 'content': sample['rejected']}]
        new_data.append({
            'chosen': chosen,
            'rejected': rejected,
            'source': 'orca_dpo_pairs'
        })
elif args.input_dataset == "Anthropic/hh-rlhf":
    for sample in dataset:
        # parse out turns and roles
        def parse_out_prompt_turns(text):
            prompt_turns = []
            text_split = text.split('Human:')
            for entry in text_split:
                if entry.strip() != '':
                    assistant_split = entry.split('Assistant:')
                    human_text = assistant_split[0].strip()
                    if len(assistant_split) > 1:
                        assistant_text = assistant_split[1].strip()
                        if human_text:
                            prompt_turns.append({"role": "user", "content": human_text})
                        if assistant_text:
                            prompt_turns.append({"role": "assistant", "content": assistant_text})
                    else:
                        if human_text:
                            prompt_turns.append({"role": "user", "content": human_text})
            return prompt_turns
        chosen_prompt_turns = parse_out_prompt_turns(sample['chosen'])
        rejected_prompt_turns = parse_out_prompt_turns(sample['rejected'])
        # run through the turns until they mismatch. This is our comparison point
        # sometimes the conversation keeps going on one, but just ignore that
        prompt_turns = []
        for i in range(min(len(chosen_prompt_turns), len(rejected_prompt_turns))):
            if chosen_prompt_turns[i] == rejected_prompt_turns[i]:
                prompt_turns.append(chosen_prompt_turns[i])
            else:
                break
        # malformed data
        if len(prompt_turns) >= len(rejected_prompt_turns):
            continue
        if len(prompt_turns) >= len(chosen_prompt_turns):
            continue
        final_chosen_turn = chosen_prompt_turns[len(prompt_turns)]
        final_rejected_turn = rejected_prompt_turns[len(prompt_turns)]
        new_data.append({
            'chosen': prompt_turns + chosen_prompt_turns,
            'rejected': prompt_turns + rejected_prompt_turns,
            'source': 'hh-rlhf'
        })
elif args.input_dataset == "lvwerra/stack-exchange-paired":
    for i, el in enumerate(dataset):
        prompt = {'content': el['question'], 'role': 'user'}
        chosen = {'content': el['response_j'], 'role': 'assistant'}
        rejected = {'content': el['response_k'], 'role': 'assistant'}
        data = {}
        data = {'chosen': [prompt, chosen], 'rejected': [prompt, rejected]}
        data['source'] = 'stack-exchange-paired'
        new_data.append(data)

# cleaning: make sure the content is always stripped
for sample in new_data:
    for msg in sample['chosen']:
        msg['content'] = msg['content'].strip()
    for msg in sample['rejected']:
        msg['content'] = msg['content'].strip()

# sanity checks over the data
# first: filter out empty content
def contains_empty(data):
    for msg in data['chosen']:
        if not msg['content']:
            return True
    for msg in data['rejected']:
        if not msg['content']:
            return True
    return False
# second: ends with assistant
def ends_with_assistant(data):
    return data['chosen'][-1]['role'] == 'assistant' and data['rejected'][-1]['role'] == 'assistant'

# apply the filters
print("Before filtering:", len(new_data))
new_data = [x for x in new_data if not contains_empty(x)]
new_data = [x for x in new_data if ends_with_assistant(x)]
print("After filtering:", len(new_data))

def _concat_messages_tulu(messages):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + "</s>" + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

# register_conv_template(
#     Conversation(
#         name="llama-2",
#         system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
#         roles=("[INST]", "[/INST]"),
#         sep_style=SeparatorStyle.LLAMA2,
#         sep=" ",
#         sep2=" </s><s>",
#     )
# )

def _concat_messages_llama_2_chat(messages):
    message_text = ""
    system_prompt = ""
    for message in messages:
        if message["role"] == "system":
            system_prompt = f"<<SYS>>\n{message['content'].strip()}<</SYS>>\n\n"
        elif message["role"] == "user":
            if len(system_prompt) > 0:
                message_text += f"<s>[INST] {system_prompt} {message['content'].strip()} [/INST]"
                system_prompt = ""
            else:
                message_text += f"<s>[INST] {message['content'].strip()} [/INST] "
        elif message["role"] == "assistant":
            message_text += f" {message['content'].strip()} </s>"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

def convert_examples(ex):
    return {
        'chosen': _concat_messages_llama_2_chat(ex['chosen']),
        'rejected': _concat_messages_llama_2_chat(ex['rejected']),
    }

new_data = [convert_examples(x) for x in new_data]

# save it
with open(args.output, 'w') as f:
    for sample in new_data:
        f.write(json.dumps(sample) + '\n')