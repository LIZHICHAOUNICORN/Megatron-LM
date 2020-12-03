#! /bin/bash
# Runs the "345M" parameter model

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=data/my-gpt2_text_document
CHECKPOINT_PATH=models

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# --load $CHECKPOINT_PATH \
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --model-parallel-size 2 \
       --num-layers 32 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file tokenizer/vocab.json \
       --merge-file tokenizer/merges.txt \
       --data-impl mmap \
       --split 998,1,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 100 \
       --fp16

set +x
