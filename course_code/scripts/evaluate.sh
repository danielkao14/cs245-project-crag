#!/bin/bash
mkdir -p log
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
MODEL_NAME=my_model_v3
K=15
EVAL_LLM=meta-llama/Llama-3.2-3B-Instruct
LLM_NAME=pretrain_models/llama3-52-peft/merged_checkpoint-480
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --model_name $MODEL_NAME \
    --k $K \
    --eval_llm_name $EVAL_LLM \
    --llm_name $LLM_NAME 2>&1 | tee log/evaluate_${MODEL_NAME}_${TIME}.log