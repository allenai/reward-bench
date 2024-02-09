import copy
import subprocess
import yaml
import random
import re
import itertools
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_eval.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/general-cirrascale"
cluster = "ai2/allennlp-cirrascale"
# cluster = "ai2/general-cirrascale-a100-80g-ib"
# cluster = "ai2/prior-elanding"
num_gpus = 1
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "high"
# d1['tasks'][0]['context']['priority'] = "preemptible"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus
# TODO: DO NOT CHECK IN
# TODO: Grab this from os.env??
d1['tasks'][0]['envVars'][6]['value'] = '---'
# TODO: enable after debugging
# TODO: add to experiment name
upload_to_hub = True
# TODO: enable to run on common preference sets instead of our custom eval set
eval_on_pref_sets = False

# python scripts/run_rm.py --model=llm-blender/PairRM-hf --direct_load
# python scripts/run_rm.py --model=berkeley-nest/Starling-RM-7B-alpha --tokenizer=meta-llama/Llama-2-7b-chat-hf --chat_template=llama-2 --direct_load --batch_size=16
# python scripts/run_rm.py --model=stanfordnlp/SteamSHP-flan-t5-xl --direct_load --batch_size=32

      # --use_slow_tokenizer # TODO: only for Llama models right now
      # --pref_sets

# """
# Parse arguments strings model and chat_template
# """
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, default="natolambert/gpt2-dummy-rm", help="path to model")
# parser.add_argument(
#     "--tokenizer", type=str, default=None, help="path to non-matching tokenizer, requires --direct_load"
# )
# parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
# parser.add_argument("--direct_load", action="store_true", help="directly load model instead of pipeline")
# parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
# parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
# parser.add_argument(
#     "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
# )
# args = parser.parse_args()
# # return args

model_configs = [
    {
        'model': 'openbmb/UltraRM-13b',
        'tokenizer': 'openbmb/UltraRM-13b',
        'chat_template': 'billa',
        'batch_size': 8,
        'direct_load': True,
    },
    {
        'model': 'OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5',
        'tokenizer': 'OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5',
        'chat_template': 'oasst_pythia',
        'batch_size': 64,
        'direct_load': True,
    },
    {
        'model': 'OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1',
        'tokenizer': 'OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1',
        'chat_template': 'oasst_pythia',
        'batch_size': 64,
        'direct_load': True,
    },
    {
        'model': 'OpenAssistant/reward-model-deberta-v3-large-v2',
        'tokenizer': 'OpenAssistant/reward-model-deberta-v3-large-v2',
        'chat_template': 'raw',
        'batch_size': 64,
        'direct_load': False,
    },
    {
        'model': 'weqweasdas/hh_rlhf_rm_open_llama_3b',
        'tokenizer': 'weqweasdas/hh_rlhf_rm_open_llama_3b',
        'chat_template': 'Robin',
        'batch_size': 64,
        'direct_load': False,
    },
    {
        'model': 'llm-blender/PairRM-hf',
        'tokenizer': 'llm-blender/PairRM-hf',
        'chat_template': 'tulu',
        'batch_size': 64,
        'direct_load': True,
    },
    {
        'model': 'berkeley-nest/Starling-RM-7B-alpha',
        'tokenizer': 'meta-llama/Llama-2-7b-chat-hf',
        'chat_template': 'llama-2',
        'batch_size': 16,
        'direct_load': True,
    },
    {
        'model': 'stanfordnlp/SteamSHP-flan-t5-xl',
        'tokenizer': 'stanfordnlp/SteamSHP-flan-t5-xl',
        'chat_template': 'tulu',
        'batch_size': 32,
        'direct_load': True,
    },
]

# model to evaluate, each in the followng format: model name, their beaker id, checkpoint subfolder
models = [
    # llama1 models
    # ("llama1-7B", "01HCCBK1MYKXKQC0C6CSVW1F22", None, "vanilla_lm"),
    # ("llama1-13B", "01HCCBWB4TWNS35N9R35K47BH8", None, "vanilla_lm"),
    # ("llama1-30B", "01HCCC7FNXFCQ2TFWGS2HA683Y", None, "vanilla_lm"),
    # ("llama1-65B", "01HCCCWQTPKS23W7MRFH5PXNHA", None, "vanilla_lm"),
    
    # llama2 models
    # ("llama2-7B", "01HCJYBBWA629B8GJTHPT496TT", None, "vanilla_lm"),
    # ("llama2-13B", "01HCJZQBM2KGQZSZRPF4HKVBZX", None, "vanilla_lm"),
    # ("llama2-70B", "01HCK281AFAXV2Y7T54NMNSC55", None, "vanilla_lm"),
    # ("llama2-chat-7B", "01HCT5D48MSRF0PCNAWNSJDN54", None, "tuned_lm"),
    # ("llama2-chat-13B", "01HCT5Q7A6FE8RZKY8TYN64ZW2", None, "tuned_lm"),
    # ("llama2-chat-70B", "01HCT63DVK7YPT6P9SN35XH417", None, "tuned_lm"),
    
    # our ablation models
    # ("finetuned_llama1_7B_dolly", "01GZVKGQZAMQMVG9307KWS4GMN", None, "tuned_lm"),
    # ("finetuned_llama1_7B_flan_v2", "01GZVKGR5DW1SXXWSMWE2QYWYR", None, "tuned_lm"),
    # ("finetuned_llama1_7B_cot", "01GZVKGRA3X4SYQF1PZ29DSZFE", None, "tuned_lm"),
    # ("finetuned_llama1_7B_code_alpaca", "01GZVKGREPDJ6FZM3S4B0J8VB9", None, "tuned_lm"),
    # ("finetuned_llama1_7B_baize", "01GZVKGRKAHJW2AK3ZF88G13HA", None, "tuned_lm"),
    # ("finetuned_llama1_7B_oasst1", "01GZVKGRQZ4359W31CAEHWFVSB", None, "tuned_lm"),
    # ("finetuned_llama1_7B_gpt4_alpaca", "01GZVKGRWJ2VVCXY5KP46814JP", None, "tuned_lm"),
    # ("finetuned_llama1_7B_super_ni", "01GZVKGS1S527GYKRA4Y26ZP5S", None, "tuned_lm"),
    # ("finetuned_llama1_7B_self_instruct", "01GZVKGS7JTYK0M35AFXHY0CD0", None, "tuned_lm"),
    # ("finetuned_llama1_7B_stanford_alpaca", "01GZVKGSHNPRFSJBS4K74FTRDC", None, "tuned_lm"),
    # ("finetuned_llama1_7B_unnatural_instructions", "01GZVKGSP9BAW8XTWB9509SPDB", None, "tuned_lm"),
    # ("finetuned_llama1_7B_sharegpt", "01GZWDNED8KP28SAR1159WZ366", None, "tuned_lm"),
    
    # ("finetuned_llama1_13B_oasst1", "01GZWN5FRTGJKEZR890MQRXZZ9", None, "tuned_lm"),
    # ("finetuned_llama1_13B_dolly", "01GZWN5FXP2ZEKJ8HBBWHK58TZ", None, "tuned_lm"),
    # ("finetuned_llama1_13B_super_ni", "01GZWN5G71CT6GFC9VC6T6RT5V", None, "tuned_lm"),
    # ("finetuned_llama1_13B_self_instruct", "01H0JSB1QDQDYPEG8AX127XMND", None, "tuned_lm"),
    # ("finetuned_llama1_13B_flan_v2", "01H04RBP7F545WC5APZK5DE58T", None, "tuned_lm"),
    # ("finetuned_llama1_13B_sharegpt", "01GZWN5G2DVDTSM508CW34V1FT", None, "tuned_lm"),
    # ("finetuned_llama1_13B_cot_lumi", "01H0F09XR3PNABMPD7X95PSR8H", None, "tuned_lm"),
    # ("finetuned_llama1_13B_baize_lumi", "01H0F123TJG9BXZ9WT42XTSDPS", None, "tuned_lm"),
    # ("finetuned_llama1_13B_code_alpaca_lumi", "01H0F1SF5WX84RXWJYZFS4CBW5", None, "tuned_lm"),
    # ("finetuned_llama1_13B_gpt4_alpaca_lumi", "01H0F43FKA2J7YY8N3K9A0CHFD", None, "tuned_lm"),
    # ("finetuned_llama1_13B_stanford_alpaca_lumi", "01H0F4TWK7YNB2YRK1TG5JEXZ5", None, "tuned_lm"),
    # ("finetuned_llama1_13B_unnatural_instructions_lumi", "01H0F5JTDM9WMKSPDBYH141089", None, "tuned_lm"),

    # ("finetuned_llama1_30B_sharegpt_lumi", "01H1SHNQXG8GSXNATQPN7GKE3T", None, "tuned_lm"),
    # ("finetuned_llama1_65B_sharegpt_lumi", "01H1SWN595ASF1NH0RBX12X96W", None, "tuned_lm"),

    # ("finetuned_llama1_7B_flanv2_cot_oasst1_dolly_lumi", "01H0K4049XMFGD8PW7BB6KVGBZ", None, "tuned_lm"),
    # ("finetuned_llama1_13B_flanv2_cot_oasst1_dolly_lumi", "01H0KJ3ZFCDBGGV4FGS8RZXCXA", None, "tuned_lm"),
    # ("finetuned_llama1_30B_flanv2_cot_oasst1_dolly_lumi", "01H0NF25QSBTVDWYV7JJNKDYCV", None, "tuned_lm"),
    # ("finetuned_llama1_65B_flanv2_cot_oasst1_dolly_lumi", "01H0P3BKSC389DSK8KBPXW8JDF", None, "tuned_lm"),
    
    # tulu v1 models
    # ("tulu_v1_7B", "01H0K6A8P9TC25F5D0NMN8NTG7", None, "tuned_lm"),
    # ("tulu_v1_13B", "01H0JW5D7ETX8252T2AHKN6S94", None, "tuned_lm"),
    # ("tulu_v1_30B", "01H0PHQSWP1CYHBYF4EG8ABX3E", None, "tuned_lm"),
    # ("tulu_v1_65B", "01H0MHPS7Y3YTND66KCP16E4AC", None, "tuned_lm"),

    # tulu v2 ablation models
    # ("finetuned_llama2_7B_on_v1_data", "01H7ABFYB84N9TN8MYXAVSMJ68", None, "tuned_lm"),
    # ("finetuned_llama2_7B_on_sharegpt", "01HEXQK5YHNWG6RW1RS1H32XXA", None, "tuned_lm"),
    # ("finetuned_llama2_13B_on_v1_data", "01H7AC0KXGRDH9ACJ24WTSK7SR", None, "tuned_lm"),
    # ("finetuned_llama2_70B_on_v1_data", "01HE9NVD58XX6G9ZYA61JZKJ7N", None, "tuned_lm"),
    # ("finetuned_llama2_7B_on_sharegpt_dpo", "01HEXR0R515HKPKTN4TNAC408A", None, "tuned_lm"),

    # tulu v2 models
    # ("tulu_v2_7B_qlora", "01HDCNBNJS56BWKP5AHV4YNCSJ", None, "tuned_lm"),
    # ("tulu_v2_13B_qlora", "01HDCNNENVNZP37VSYR3AZSMYT", None, "tuned_lm"),
    # ("tulu_v2_70B_qlora", "01HDG3YXJD6TKNFW6WV19NE7A0", None, "tuned_lm"),
    # ("tulu_v2_7B_jax", "01HBXTF305QARZ7P4T6ASXXVAM", None, "tuned_lm"),
    # ("tulu_v2_13B_jax", "01HBWE5NHC3M30HH63339HS8BE", None, "tuned_lm"),
    # ("tulu_v2_70B_jax", "01HCB2VZJ2T2JXZX0R1SJBRSB2", None, "tuned_lm"),
    # ("tulu_v2_7B_dpo", "01HE8H1MBSVN09ZZ82X6K90NTF", None, "tuend_lm"),
    # ("tulu_v2_13B_dpo", "01HE8YMBMJSTJV49QWA6TF2NTE", None, "tuend_lm"),
    # ("tulu_v2_70B_dpo_first_epoch", "01HES1TCSJCPTPV50HQZHSN319", None, "tuend_lm"),
    # ("tulu_v2_70B_dpo_second_epoch", "/net/nfs.cirrascale/allennlp/hamishi/EasyLM/tulu_2_70b_dpo/", None, "tuend_lm"),
    # ("tulu_v2_70B_dpo", "01HEXKXP0MFM60PT7SY71XXSWD", None, "tuend_lm"),

    # code llama models
    # ("code_llama_7B", "01HD9Z1MJ9K3ZK494KGTVD1063", None, "vanilla_lm"),
    # ("code_llama_13B", "01HD9Z9TNEFWS5E8MQJMDY6N0P", None, "vanilla_lm"),
    # ("code_llama_34B", "01HD9ZQF6PRAMC0ANVPFJFEJHR", None, "vanilla_lm"),
    # ("code_llama_instruct_7B", "01HDA0SGJ0GB2ZF6D6RXS6NREZ", None, "tuned_lm"),
    # ("code_llama_instruct_13B", "01HDA141K4SEDPXFY749092FNZ", None, "tuned_lm"),
    # ("code_llama_instruct_34B", "01HDA1GNSCCNDQ4FNQ2FPPRBSD", None, "tuned_lm"),
    
    # code tulu models
    # ("code_tulu_7B_jax", "01HD57SA48PBKD30FKB2F55S7H", None, "tuned_lm"),
    # ("code_tulu_13B_jax", "01HCTQG860G68C2486K1QNSY3S", None, "tuned_lm"),
    # ("code_tulu_34B_jax", "01HD7J73FJ7299VQKPKBS8RSJB", None, "tuned_lm"),


    # other causal models
    # ("hf-opt-7B", "facebook/opt-6.7b", None, "vanilla_lm"),
    # ("finetuned_opt_7B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca", "01H13EBXSADXXJCRERART90ZKJ", None, "tuned_lm"),
    # ("hf-pythia-7B", "EleutherAI/pythia-6.9b", None, "vanilla_lm"),
    # ("fintuned_pythia_7B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca", "01H1359QTQZCXFTW4KY4WVKF0C", None, "tuned_lm"),
    # ("hf-falcon-40B", "tiiuae/falcon-40b", None, "vanilla_lm"),
    # ("finetuned_falcon_40B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca", "01H2TRXD9TE80W61PABE26785P", None, "tuned_lm"),
    # ("hf-falcon-7B", "tiiuae/falcon-7b", None, "vanilla_lm"),
    # ("finetuned_falcon_7B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca", "01H356X9ZYY8HX1C7HFH6JYWNW", None, "tuned_lm"),
    # ("hf-falcon-rw-7B", "tiiuae/falcon-rw-7b", None, "vanilla_lm"),
    # ("finetuned_falcon_rw_7B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca", "01H37QXWFK095588W6GCMVGFKB", None, "tuned_lm"),
    # ("zephyr-7B", "/net/nfs.cirrascale/allennlp/yizhongw/checkpoints/zephyr-7b-beta", None, "tuned_lm"),
    # ("xwin-70B", "/net/nfs.cirrascale/allennlp/yizhongw/checkpoints/Xwin-LM-70B-V0.1", None, "tuned_lm"),
]

#--------------- experiments about number of supervision tasks -------------------------

# for experiment_group, model_info in itertools.product(experiment_groups, models):
# for model_info, experiment_group in itertools.product(models, experiment_groups):
for model_config in model_configs:
    if eval_on_pref_sets:
        experiment_group = 'common-preference-sets'
    else:
        experiment_group = 'herm-preference-sets'
    print(f"Submitting evaluation for model: {model_config['model']} on {experiment_group}")
    d = copy.deepcopy(d1)

    # name = f"open_instruct_eval_{experiment_group}_{model_name}_{today}"
    name = f'herm_eval_for_{model_config["model"]}_on_{experiment_group}'.replace('/', '-')
    d['description'] = name
    d['tasks'][0]['name'] = name

    d['tasks'][0]['arguments'][0] = (f"python scripts/run_rm.py"
        f" --model {model_config['model']}"
        f" --tokenizer {model_config['tokenizer']}"
        f" --chat_template {model_config['chat_template']}"
        f" --batch_size {model_config['batch_size']}")
    
    if model_config['direct_load']:
        d['tasks'][0]['arguments'][0] += ' --direct_load'
    if not upload_to_hub:
        d['tasks'][0]['arguments'][0] += ' --do_not_save'
    if eval_on_pref_sets:
        d['tasks'][0]['arguments'][0] += ' --pref_sets'
        # --direct_load
        # --do_not_save
        # --use_slow_tokenizer # TODO: only for Llama models right now
        # --pref_sets



    #     python -m eval.mmlu.run_eval \
    #     --ntrain 0 \
    #     --data_dir /data/mmlu/ \
    #     --save_dir /output/ \
    #     --model_name_or_path /model \
    #     --tokenizer_name_or_path /model \
    #     --eval_batch_size 4 \
    #     --load_in_8bit \
    #     --use_chat_format \
    #     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
    # '''

    # if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
    # if model_info[1].startswith("/"):  # if it's a local model, load it from the local directory
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
    # else:  # if it's a beaker model, mount the beaker dataset to `/model`
    #     d['tasks'][0]['datasets'][1]['source']['beaker'] = model_info[1]

    # # if a specific checkpoint is specified, load model from that checkpoint
    # if model_info[2] is not None:
    #     # extract existing model path
    #     model_name_or_path = re.search("--model_name_or_path (\S+)", d['tasks'][0]['arguments'][0]).group(1)
    #     # replace the model path with the checkpoint subfolder
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(model_name_or_path, model_name_or_path+"/"+model_info[2])]
    #     # replace the tokenizer path with the checkpoint subfolder
    #     tokenizer_name_or_path = re.search("--tokenizer_name_or_path (\S+)", d['tasks'][0]['arguments'][0]).group(1)

    # # for vanilla_lm, remove the chat formatting function
    # if model_info[3] == "vanilla_lm":
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]

    # if "13B" in model_info[0]:
    #     # find the batch size argument, and reduce by 4x
    #     if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
    #         original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
    #         new_batch_size = max(1, int(original_batch_size) // 2)
    #         d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]


    # if "30B" in model_info[0] or "34B" in model_info[0]:
    #     # find the batch size argument, and reduce by 4x
    #     if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
    #         original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
    #         new_batch_size = max(1, int(original_batch_size) // 4)
    #         d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

    #     if "codex_eval" in experiment_group:
    #         # request 2x more GPUs
    #         d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']
    
    # elif "70B" in model_info[0] or "65B" in model_info[0] or "40B" in model_info[0]:
    #     # find the batch size argument, and reduce by 4x
    #     if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
    #         original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
    #         new_batch_size = max(1, int(original_batch_size) // 4)
    #         d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

    #     if "codex_eval" in experiment_group:
    #         # request 4x more GPUs
    #         d['tasks'][0]['resources']['gpuCount'] = 4 * d['tasks'][0]['resources']['gpuCount']
    #     else:
    #         # request 2x more GPUs
    #         d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']


    # if "llama2-chat" in model_info[0]:
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
    #         "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
    #         "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
    #     ]
    # elif "code_llama_instruct" in model_info[0]:
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
    #         "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
    #         "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
    #     ]
    # elif "zephyr" in model_info[0]:
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
    #         "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
    #         "--chat_formatting_function eval.templates.create_prompt_with_zephyr_chat_format")
    #     ]
    # elif "xwin" in model_info[0]:
    #     d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
    #         "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
    #         "--chat_formatting_function eval.templates.create_prompt_with_xwin_chat_format")
    #     ]

    # if any([x in model_info[0] for x in ["opt", "pythia", "falcon"]]):
    #     if "--use_vllm" in d['tasks'][0]['arguments'][0]:
    #         print(f"Removing --use_vllm for {model_info[0]}")
    #         d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_vllm", "")] 

    # print(d)

    fn = "beaker_configs/auto_created/{}.yaml".format(name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/herm".format(fn)
    subprocess.Popen(cmd, shell=True)