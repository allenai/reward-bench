# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

MODEL_SIZE=7B
# BATCH_SIZE_PER_GPU=1
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
# TOTAL_BATCH_SIZE=32 # try smaller batch for 13b
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# OUTPUT_DIR=/net/nfs.cirrascale/allennlp/jacobm/herm/rms/ultrafeedback/test-tinyllama-ultrafeedback-repro-uf-settings
MODEL_PATH=allenai/tulu-2-7b
# OUTPUT_DIR=/net/nfs.cirrascale/allennlp/jacobm/herm/rms/ultrafeedback/tulu-2-7b-ultrafeedback-repro-1e-5-linear
# OUTPUT_DIR=/net/nfs.cirrascale/allennlp/jacobm/herm/rms/nectar/tulu-2-7b-nectar-binarized-filtered-1e-5-linear-lora
OUTPUT_DIR=/net/nfs.cirrascale/allennlp/jacobm/herm/rms/nectar/lora-tulu-2-7b-nectar-binarized-filtered-debug
# MODEL_PATH=allenai/tulu-2-13b
# OUTPUT_DIR=/net/nfs.cirrascale/allennlp/jacobm/herm/rms/ultrafeedback/debug-tulu-2-13b-ultrafeedback-repro-tulu-settings
# TRAIN_DATASET=ultrafeedback
# TRAIN_DATASET=nectar
TRAIN_DATASET=nectar-binarized-filtered
# EVAL_DATASET=alpaca_farm_human_preferences
# TRAIN_DATASET=Anthropic/hh-rlhf
# TRAIN_DATASET=Dahoas/synthetic-instruct-gptj-pairwise
# OUTPUT_DIR=models/llama-2-7b-gptj-pairwise-5-epochs
# OUTPUT_DIR=net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/${DATASET}_${MODEL_SIZE}/
echo "Training model ${MODEL_PATH} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed --include localhost:0,1,2,3 scripts/train_rm_trainer.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name $MODEL_PATH \
    --dataset_name $TRAIN_DATASET \
    --max_seq_length 1024 \
    --preprocessing_num_workers 64 \
    --do_train \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --bf16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
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
    # \
    # --gradient_checkpointing

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

# enable after debugging
# python scripts/merge_lora.py \
#     --base_model_name_or_path $MODEL_PATH \
#     --lora_model_name_or_path $OUTPUT_DIR \
#     --output_dir {$OUTPUT_DIR}_merged \
#     --save_tokenizer