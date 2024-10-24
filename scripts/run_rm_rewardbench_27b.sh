export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/cephfs/zhangge/root/code/reward-bench:$PYTHONPATH"

# 定义模型目录路径
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau1_no_sft_2e6_sky80k_cleaned_bt_epoch2"  # 请替换为实际的模型目录路径
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau01_no_sft_2e6_sky80k_cleaned_epoch2_vh2_w_moe_w_l2"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau01_no_sft_2e6_sky80k_cleaned_epoch2_vh4_w_moe_w_l2"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau01_no_sft_2e6_sky80k_cleaned_epoch2_vh6_w_moe_w_l2"
model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau01_no_sft_2e6_sky80k_cleaned_epoch2_vh8_w_moe_w_l2"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau01_no_sft_1e5_sky80k_cleaned_epoch1_vh2_w_moe_w_l2"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau01_no_sft_2e6_sky80k_cleaned_epoch1_vh2_w_moe_w_l2"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-27b-it/rm/batch32_tau01_no_sft_5e7_sky80k_cleaned_epoch1_vh2_w_moe_w_l2"

# 获取模型名称
model_name=$(basename "$model_dir")

# 定义日志文件路径
log_file="$model_dir/run_log.txt"


# 确定额外参数
if [[ $model_name =~ vh ]]; then
    extra_params="--is_general_preference"
else
    extra_params=""
fi


# 设置batch_size
batch_size=32

echo "Running model: $model_name. Log file: $log_file"

# 运行模型并将输出重定向到日志文件
(
    echo "Starting run for model: $model_name at $(date)"
    python run_rm.py \
    --model "$model_dir" \
    --chat_template raw \
    --batch_size $batch_size \
    --max_length 4096 \
    --do_not_save \
    --torch_dtype "bfloat16" \
    --trust_remote_code \
    --attn_implementation "flash_attention_2" \
    $extra_params
    echo "Finished run for model: $model_name at $(date)"
) > "$log_file" 2>&1

echo "Model run completed. Log file: $log_file"