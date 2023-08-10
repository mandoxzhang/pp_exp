#! /bin/bash

# Runs the "345M" parameter model


# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DATA_PATH='./'
CHECKPOINT_PATH=./

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# GPT-3 Small 125M
# MODEL_SIZE=0.125
# NUM_LAYERS=12
# HIDDEN_SIZE=768
# NUM_ATTN_HEADS=12
# GLOBAL_BATCH_SIZE=256
# LR=6.0e-4
# MIN_LR=6.0e-5

## GPT-3 XL 1.3B
MODEL_SIZE=1.3
NUM_LAYERS=6
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
GLOBAL_BATCH_SIZE=512
LR=2.0e-4
MIN_LR=2.0e-5

# GPT-3 2.7B
# MODEL_SIZE=2.7
# NUM_LAYERS=32
# HIDDEN_SIZE=2560
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=512
# LR=1.6e-4
# MIN_LR=1.6e-5

# # GPT-3 6.7B
# MODEL_SIZE=6.7
# NUM_LAYERS=32
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=1024
# LR=1.2e-4
# MIN_LR=1.2e-5

# GPT-3 13B
# MODEL_SIZE=13
# NUM_LAYERS=40
# HIDDEN_SIZE=5120
# NUM_ATTN_HEADS=40
# GLOBAL_BATCH_SIZE=1024
# LR=1.0e-4
# MIN_LR=1.0e-5

# GPT-3 30B
# MODEL_SIZE=13
# NUM_LAYERS=60
# HIDDEN_SIZE=6640
# NUM_ATTN_HEADS=40
# GLOBAL_BATCH_SIZE=1024
# LR=1.0e-4
# MIN_LR=1.0e-5
# python -m torch.distributed.launch $DISTRIBUTED_ARGS \
TP_SIZE=1
PP_SIZE=1
# EP_SIZE=8
# NUM_GPUS=8
EP_SIZE=1
NUM_GPUS=1
if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi

MLC=0.01
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"

megatron_options=" \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size $PP_SIZE \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP}"

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

# if [[ $EP_SIZE -gt 1 ]]; then
# deepspeed_options="${deepspeed_options} \
#         --no-pipeline-parallel"
# fi
# export NCCL_PROTOS=2
# # export NCCL_DEBUG=TRACE
# export NCCL_IB_HCA=^mlx5_3:1
# export NCCL_SOCKET_IFNAME=bond0:10.11.1.2

deepspeed --hostfile ./hostfile \
       pretrain_llama.py \
       ${megatron_options} \
       --deepspeed \
       --deepspeed_config ./deepspeed.json \
       --no-pipeline-parallel \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTN_HEADS \
       --micro-batch-size 1 \
       --seq-length 1023 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file gpt2-vocab.json \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend mccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --zero-stage 3 \
       --no-bias-dropout-fusion \
# --cpu-optimizer

#bash pretrain_gpt_distributed.sh 1>mccl_log/$(date "+%Y-%m-%d_%H:%M:%S").txt 2> error_log/$(date "+%Y-%m-%d_%H:%M:%S").txt