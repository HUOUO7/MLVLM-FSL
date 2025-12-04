#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=2
DIR=`pwd`
export CUDA_VISIBLE_DEVICES='0,7'
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat-Int4" # Qwen/Qwen-VL-Chat-Int4 Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
#DATA="/home/user4/cww/Elevater_Toolkit_IC-main/output7_newformat_Qwen_cww/base/700_Mini-ImageNet_base_5way1shot1query.json"
DATA="/data/user4/cww/Data_Construction/summary_experiments/tune_elevater13_1w6_data/generate_base/lunwen_prompt/output7_newformat_Qwen_finetune_13_elevater_add_answerlist_QandA_base_get_ratio_data_summary_orginal.json"

#DATA='/data/user4/cww/Data_Construction/output7_newformat_Qwen_finetune_13_elevater_add_answerlist_QandA/output7_newformat_Qwen_finetune_13_elevater_add_answerlist_QandA_base_get_ratio_data.json'


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Remember to use --fp16 instead of --bf16 due to autogptq
torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --fp16 True \
    --fix_vit True \
    --output_dir output_qwen/summary_experiments/tune_elevater13_1w6_summary_orginal \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --use_lora \
    --q_lora \
    --gradient_checkpointing \
    --deepspeed finetune/ds_config_zero2.json