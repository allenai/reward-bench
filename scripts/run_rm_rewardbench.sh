export CUDA_VISIBLE_DEVICES=0
python run_rm.py \
--model general-preference/GPM-Gemma-2B \
--chat_template raw \
--batch_size 64 \
--max_length 4096 \
--do_not_save \
--trust_remote_code \
--attn_implementation "flash_attention_2" \
--is_general_preference 


# general-preference/GPM-Llama-3.1-8B-Instruct
# general-preference/GPM-Gemma-2B
# general-preference/GPM-Gemma-2-2B
# general-preference/GPM-Gemma-2-9B

