# CS 245 Final Project Group 1

# Table of Contents

1. [Environment Setup](#environment-setup)
2. [Download Dataset](#download-dataset)
3. [Download dehallucinated model weight](#download-dehallucinated-model-weight)
4. [Reproducing results](#reproducing-results)

# Environment Setup

We used `conda` to set up our environment.
```bash
conda create -n crag python==3.11
conda activate crag
pip install -r requirements.txt
```

# Download Dataset
Download the dataset for Task #1 at the following [link](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files). Your `data` folder should look like this.

```
data
├── crag_task_1_dev_v4_release.jsonl.bz2
```

# Download Dehallucinated Model Weight

We used the fine-tuned model weight at the following [link](https://gitlab.aicrowd.com/jiazunchen/kdd2024cup-crag-db3/-/tree/main/models/pretrain_models/llama3-52-peft/checkpoint-480) but merged the LoRA adapter to the base model so that we could run vLLM inference more easily. 

```bash
cd course_code
mkdir pretrain_models
cd pretrain_models
pip install gdown
gdown 1aiGNzXmx18D5u3Nog3Ef4TflHJBsDBQI
```

If `gdown` doesn't work, consider downloading directly at the following GDrive [link](https://drive.google.com/file/d/1aiGNzXmx18D5u3Nog3Ef4TflHJBsDBQI/view?usp=sharing).

# Reproducing Results

## Generation
```bash
cd course_code
./scripts/generate.sh
```

- `MODEL_NAME`: method name, choices = [`vanilla_baseline`, `rag_baseline`, `my_model_v1`, `my_model_v2`, `my_model_v3`]. The final method is `my_model_v3`
- `LLM_NAME`: model path, either provide a valid HuggingFace model or a local model path. For the de-hallucinated model, use `pretrain_models/merged_checkpoint-480`.
- `K`: number of retrieved chunks for Parent-Child Chunk retriever.

## Evaluation
```bash
cd course_code
./scripts/evaluate.sh
```

- `MODEL_NAME`: method name, choices = [`vanilla_baseline`, `rag_baseline`, `my_model_v1`, `my_model_v2`, `my_model_v3`]. The final method is `my_model_v3`
- `EVAL_LLM`: evaluation model path, either provide a valid HuggingFace model or a local model path. Note that we fixed it to `meta-llama/Llama-3.2-3B-Instruct` for a fair comparison.
- `LLM_NAME`: generation model path, either provide a valid HuggingFace model or a local model path. For the de-hallucinated model, use `pretrain_models/merged_checkpoint-480`.
- `K`: number of retrieved chunks for Parent-Child Chunk retriever.