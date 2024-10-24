#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="/cephfs/zhangge/root/code/reward-bench:$PYTHONPATH"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/Llama-3-8B-Instruct/rm"  # 请替换为实际的模型目录路径
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-9b-it/rm"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2-2b-it/rm"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2b-it/rm"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/Llama-3-8B-Instruct/rm_ablation"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/Llama-3-8B-Instruct/rm_ablation_no_l2"
# model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2b-it/rm_ablation"
model_dir="/cephfs/zhangge/root/code/sqz_general_preference/results/saved_model/gemma-2b-it/rm_ablation_no_l2"
# 获取可用的 GPU 数量
gpu_count=$(nvidia-smi -L | wc -l)

# 获取模型目录列表
model_dirs=()
while IFS=  read -r -d $'\0'; do
    model_dirs+=("$REPLY")
done < <(find "$model_dir" -mindepth 1 -maxdepth 1 -type d -print0)

# 计算模型数量
model_count=${#model_dirs[@]}

run_models() {
    local start=$1
    local end=$2
    
    for ((i=start; i<end; i++)); do
        model_path="${model_dirs[i]}"
        model_name=$(basename "$model_path")
        log_file="$model_path/run_log.txt"
        
        # 提取 value_head_dim
        if [[ $model_name =~ vh([0-9]+) ]]; then
            value_head_dim="${BASH_REMATCH[1]}"
        else
            value_head_dim="1"
        fi
        
        # # 确定额外参数和batch_size
        # if [[ $model_name =~ vh ]]; then
        #     if [[ $model_name =~ w_moe ]]; then
        #         extra_params="--is_general_preference --add_prompt_head"
        #     elif [[ $model_name =~ no_moe ]]; then
        #         extra_params="--is_general_preference"
        #     else
        #         extra_params=""
        #     fi
        # else
        #     extra_params=""
        # fi
        extra_params=""

        # 设置batch_size
        if [[ $model_name =~ 27b ]]; then
            batch_size=32
        else
            batch_size=64
        fi
        
        echo "Running model: $model_name on GPU $i. Log file: $log_file"
        (
            echo "Starting run for model: $model_name on GPU $i at $(date)"
            CUDA_VISIBLE_DEVICES=$i python run_rm.py \
            --model $model_path \
            --chat_template raw \
            --batch_size $batch_size \
            --value_head_dim $value_head_dim \
            --max_length 4096 \
            --do_not_save \
            --torch_dtype "bfloat16" \
            --trust_remote_code \
            --attn_implementation "flash_attention_2" \
            $extra_params
            echo "Finished run for model: $model_name on GPU $i at $(date)"
        ) > "$log_file" 2>&1 &
    done
    
    wait
}

# 运行第一批模型
first_batch_size=$((model_count < gpu_count ? model_count : gpu_count))
run_models 0 $first_batch_size

# 如果还有剩余模型，运行第二批
if [ $model_count -gt $gpu_count ]; then
    echo "Running remaining models..."
    run_models $gpu_count $model_count
fi

echo "All model runs completed."