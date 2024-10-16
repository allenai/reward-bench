export CUDA_VISIBLE_DEVICES=0
python run_rm.py \
--model general-preference/GPM-Gemma-2B \
--chat_template raw \
--batch_size 64 \
--value_head_dim 8 \
--max_length 4096 \
--do_not_save \
--trust_remote_code \
--attn_implementation "flash_attention_2" \
--is_general_preference \
--add_prompt_head 


# general-preference/GPM-Llama-3.1-8B
# general-preference/GPM-Gemma-2B

