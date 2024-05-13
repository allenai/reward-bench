import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List


class SlicPairPMPipeline:

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


