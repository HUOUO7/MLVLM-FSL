export CUDA_VISIBLE_DEVICES='1'

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=25646

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

#checkpoint="/data/user4/HUOJIAN/Qwen-VL/output_qwen/fsl_13_elevater_10w_pseudo_QandA/checkpoint-4500"
checkpoint="/data/user4/HUOJIAN/Qwen-VL/output_qwen/fsl_13_elevater_10w_QandA/checkpoint-3500"
#evaluate_config_path="/data/user4/HUOJIAN/Qwen-VL/eval_mm/FSL_eval_config/benchmark_1.json"
evaluate_config_path="/data/user4/HUOJIAN/Qwen-VL/eval_mm/FSL_eval_config/0807_vani.json"
evaluate_save_folder="/data/user4/HUOJIAN/Qwen-VL/cww_test/output_qwen_5000_balance/0807_regenerate_vani3500-1"
ref_history_path="/data/user4/cww/Data_Construction/output7_newformat_Qwen_cww/base_history"

python -m torch.distributed.launch --use-env $DISTRIBUTED_ARGS eval_mm/evaluate_for_FSL_v1_no_history.py \
    --checkpoint $checkpoint \
    --evaluate_config_path $evaluate_config_path \
    --evaluate_save_folder $evaluate_save_folder \
    --ref_history_path $ref_history_path \
    --batch-size 4 \
    --num-workers 4