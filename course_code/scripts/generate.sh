#!/bin/bash
mkdir -p log
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
MODEL_NAME=my_model_v3
K=5
LLM_NAME=pretrain_models/merged_checkpoint-480
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --split 1 \
    --model_name $MODEL_NAME \
    --k $K \
    --llm_name $LLM_NAME 2>&1 | tee log/generate_${MODEL_NAME}_${TIME}.log