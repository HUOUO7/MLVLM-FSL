export CUDA_VISIBLE_DEVICES='4,3'

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=25625

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

checkpoint="/data/user4/HUOJIAN/Qwen-VL/Qwen/Qwen-VL-Chat-Int4"

# You can organize your own test data sets(evaluate_config_path)
evaluate_config_path="/data/user4/HUOJIAN/Qwen-VL/eval_mm/FSL_eval_config/0807_NOTUNE.json"

# The output path of the evaluation results(evaluate_save_folder)
evaluate_save_folder="/data/user4/HUOJIAN/Qwen-VL/cww_test/output_qwen_5000_balance/0807_regenerate_NOTUNE"

# Historical dialogue file, added by historical dialogue experiment, later proved useless, can be left alone(can ignored)
# If you want to try it out, go to evaluate_for_FSL_notune.py and change the history-related code
ref_history_path="/data/user4/cww/Data_Construction/output7_newformat_Qwen_cww/base_history"

python -m torch.distributed.launch --use-env $DISTRIBUTED_ARGS eval_mm/evaluate_for_FSL_notune.py \
    --checkpoint $checkpoint \
    --evaluate_config_path $evaluate_config_path \
    --evaluate_save_folder $evaluate_save_folder \
    --ref_history_path $ref_history_path \
    --batch-size 4 \
    --num-workers 4
