export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=8
# TOTAL_BATCH_SIZE=512
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# MODEL_PATH=openai-community/gpt2-large
# MODEL_PATH=TinyLlama/TinyLlama-1.1B-step-50K-105b
# MODEL_PATH=/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B
MODEL_PATH=mistralai/Mixtral-8x7B-v0.1
# TRAIN_DATASET=lvwerra/stack-exchange-paired
# EVAL_DATASET=lvwerra/stack-exchange-paired
# TRAIN_DATASET=mychen76/stack-exchange-paired-500k
# EVAL_DATASET=mychen76/stack-exchange-paired-500k
TRAIN_DATASET=alpaca_farm_human_preferences
# EVAL_DATASET=alpaca_farm_human_preferences
# TRAIN_DATASET=Anthropic/hh-rlhf
# TRAIN_DATASET=Dahoas/synthetic-instruct-gptj-pairwise
# OUTPUT_DIR=models/llama-2-7b-gptj-pairwise-5-epochs
# OUTPUT_DIR=net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/${DATASET}_${MODEL_SIZE}/
OUTPUT_DIR=models/test-mixtral-alpaca-farm-1-epoch
echo "Training model ${MODEL_PATH} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed --include localhost:0,1,2,3,4,5,6,7 scripts/train_rm_trainer.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --dataset_name $TRAIN_DATASET \
    --max_seq_length 512 \
    --preprocessing_num_workers 16 \
    --do_train \
    --use_flash_attn \
    --bf16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy no \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --num_train_epochs 1 \
    --output_dir $OUTPUT_DIR \
    --use_slow_tokenizer \
    --overwrite_output_dir
    # --bf16_full_eval \
    # --torch_dtype bfloat16 \
    # --do_eval \

    # --use_slow_tokenizer \
    # --output_dir output/tulu_v1_${MODEL_SIZE}/ # \
    # --bf16 \
    # --tf32 True \
    # --torch_dtype bfloat16 \
    # --overwrite_output_dir # \
    # --report_to "tensorboard" \
    # --max_steps 10 