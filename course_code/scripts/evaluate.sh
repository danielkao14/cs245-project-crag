#!/bin/bash
mkdir -p log
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
MODEL_NAME=vanilla_baseline
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --model_name $MODEL_NAME \
    --llm_name "meta-llama/Meta-Llama-3-8B-Instruct" 2>&1 | tee log/evaluate_${MODEL_NAME}_${TIME}.log