#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" \
    --split 1 \
    --model_name "my_model" \
    --llm_name "meta-llama/Llama-3.2-3B-Instruct"