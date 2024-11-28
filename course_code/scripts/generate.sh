#!/bin/bash
mkdir log
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
CUDA_VISIBLE_DEVICES=0,1 python generate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --split 1 \
    --model_name "my_model_v1" \
    --llm_name "meta-llama/Llama-3.2-3B-Instruct" 2>&1 | tee log/generate_${TIME}.log