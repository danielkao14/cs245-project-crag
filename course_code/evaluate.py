import bz2
import json
import os
from datetime import datetime
import argparse

from loguru import logger
from tqdm.auto import tqdm

import vllm
from openai import OpenAI



def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        if "true" in resp:
            answer = 1

        return answer
    except:
        return -1


def evaluate_predictions(results, eval_model):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    queries, ground_truths, predictions = results["queries"], results["ground_truths"], results["predictions"]

    llm_evaluation_logs = [] # record queries that need llm evaluation

    for _idx, prediction in tqdm(enumerate(predictions), total=len(predictions)):
        query = queries[_idx]
        ground_truth = str(ground_truths[_idx]).strip()
        prediction = prediction.strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        if "i don't know" in prediction_lowercase:
            n_miss += 1
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            continue
        # else use llm evaluator to eval
        response = eval_model.evaluate(query, ground_truth, prediction)
        llm_evaluation_logs.append({"query": query, "ground_truth": ground_truth, "prediction": prediction, "response": response})
        eval_res = parse_response(response)
        if eval_res == 1:
            n_correct += 1

    n = len(predictions)
    evaluation_results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(evaluation_results)
    return evaluation_results, llm_evaluation_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="example_data/dev_data.jsonl.bz2",
                        choices=["example_data/dev_data.jsonl.bz2", # example data
                                 "data/crag_task_1_dev_v4_release.jsonl.bz2", # full data
                                 ])

    parser.add_argument("--model_name", type=str, default="vanilla_baseline",
                        choices=["vanilla_baseline",
                                 "rag_baseline",
                                 # add your model here
                                 "my_model",
                                 "my_model_v1",
                                 "my_model_v2"
                                 ],
                        )

    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        choices=["meta-llama/Llama-3.2-3B-Instruct",
                                 "google/gemma-2-2b-it",
                                 # can add more llm models here
                                 "meta-llama/Meta-Llama-3-8B-Instruct",
                                 ])
    parser.add_argument("--is_server", action="store_true", default=False,
                        help="Whether we use vLLM deployed on a server or offline inference.")
    parser.add_argument("--vllm_server", type=str, default="http://localhost:8088/v1",
                        help="URL of the vLLM server if is_server is True. The port number may vary.")
    parser.add_argument("--max_retries", type=int, default=10,
                        help="Number of retries for evaluation per query.")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of retrieved documents when using ParentDocumentRetriever")


    args = parser.parse_args()
    print(args.is_server)

    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[0]
    dataset_path = os.path.join("..", dataset_path)

    llm_name = args.llm_name
    _llm_name = llm_name.split("/")[-1]

    # init evaluation model
    from evaluation_model import EvaluationModel
    eval_model = EvaluationModel(llm_name=llm_name, is_server=args.is_server,
                                 vllm_server=args.vllm_server, max_retries=args.max_retries)


    # get output directory
    model_name = args.model_name
    output_directory = os.path.join("..", "output", dataset, model_name, _llm_name)
    if model_name == "my_model_v2":
        output_directory = os.path.join(output_directory, f"k={args.k}")
    if not os.path.exists(output_directory):
        raise FileNotFoundError(f"Output directory {output_directory} does not exist.")

    # load predictions
    predictions_file = os.path.join(output_directory, "predictions.json")
    results = json.load(open(predictions_file))

    # Evaluate predictions
    evaluation_results, llm_evaluation_logs = evaluate_predictions(results, eval_model)

    # save evaluation_results
    json.dump(evaluation_results, open(os.path.join(output_directory, "evaluation_results.json"), "w"), indent=4)
    json.dump(llm_evaluation_logs, open(os.path.join(output_directory, "llm_evaluation_logs.json"), "w"), indent=4)
