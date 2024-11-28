import os
import time
from typing import Any, Dict, List

import ray
import torch
import vllm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm

from retriever import Retriever
from templates import template_map


#### CONFIG PARAMETERS ---
# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
#### CONFIG PARAMETERS END---

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(
        self, 
        llm_name="meta-llama/Meta-Llama-3-8B-Instruct", # Llama-3.2-3B-Instruct
        is_server=False, 
        vllm_server=None
    ):
        self.initialize_models(llm_name, is_server, vllm_server)
        # self.used1 = "cuda:0"
        # self.used2 = "cuda:1"
        self.Task = 1
        self.k = 15

    def initialize_models(self, llm_name, is_server, vllm_server):
        # self.llm_name = llm_name
        self.llm_name = "/home/dangnth/cs245-project-crag/course_code/pretrain_models/llama3-52-peft/merged_checkpoint-480"
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True,
            )
            self.used1 = "cuda:0"
            self.used2 = "cuda:1"
            self.used = "cuda:0"

            self.tokenizer = self.llm.get_tokenizer()
            
        # Initialize the retriever
        self.retriever = Retriever(
            token_path=self.llm_name,
            device1=self.used1, 
            device2=self.used2,
            batch_size=64,             
            hf_path="sentence-transformers/all-MiniLM-L6-v2", 
            bge_large_path="BAAI/bge-base-en-v1.5", 
            parent_chunk_size=700, 
            parent_chunk_overlap=150,
            child_chunk_size=200, 
            child_chunk_overlap=50,
            rerank_path='BAAI/bge-reranker-v2-m3'
        )
        
        self.retriever.clear()
        
    
    def llama3_domain(self, query):
        """
        Predict query domain

        Args:
            query (str): Query

        Returns:
            str: Domain. Choices are [finance, music, sports, movie, open]
        """
        messages = [
            {
                "role": "system", 
                "content": f"You are an assistant expert in movie, sports, finance and music fields."
            },
            {
                "role": "user",
                "content": "Please judge which category the query belongs to, without answering the query. you can only and must output one word in (movie, sports, finance, music) If the question doesn't belong to movie, sports,finance, music, please answer other. \n Query:" + query + '\n Category:'
            },
        ]
        
        domain = self.llama3_output(messages, maxtoken=3)
        
        for key in ['finance', 'music', 'sports', 'movie']:
            if key in domain:
                return key
        
        return 'open'
    
    def llama3_output(self, messages, maxtoken=75):
        """
        Generate output using LLama3

        Args:
            messages (List[str]): A list of messages for which answers are generated.
            maxtoken (str): Maximum number of generated tokens

        Returns:
            str: Model output
        """
        # Check if time out
        if time.time() - self.all_st >= self.all_time:
            return "i don't know"
            
        # Generate responses via vllm
        # note that here self.batch_size = 1
        t1 = time.time()
        # Convert to chat template
        messages = self.format_chat(messages)
        # print(f"Messages: {messages}")
        
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=maxtoken,  # Maximum number of tokens to generate per output sequence.
            )
            output = response.choices[0].message.content
        else:
            # terminators = [
            #     self.tokenizer.eos_token_id,
            #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            # ]

            responses = self.llm.generate(
                messages,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=maxtoken,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            output = responses[0].outputs[0].text
                
        # print("end gen:", time.time() - t1)
        print("output:", output)

        return output

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size
    
    def process_task1(self, domain, query, query_time):
        context_str = ""
        output = ""
        
        if domain in ['movie']:
            context_str = self.retriever.get_movie_oscar(query)
            if context_str is not None:
                t1 = time.time()
                filled_template = template_map['output_answer_nofalse'].format(
                    context_str=context_str,
                    query_str=query)
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"
                    },
                    {
                        "role": "user", 
                        "content": filled_template
                    },
                ]
                output = self.llama3_output(messages, maxtoken=70)
                # print("end oscar:", time.time() - t1)
                if "i don't know" not in output and "invalid" not in output:
                    return output, context_str
            else:
                context_str = ""
                t1 = time.time()
                filled_template = template_map['ask_name'].format(query_str=query)
                messages = [
                    {
                        "role": "system",
                        "content": f" You will be asked a lot of questions, but you don't need to answer them, just point out the name of the movie involved."
                    },
                    {
                        "role": "user", 
                        "content": filled_template
                    },
                ]
                output = self.llama3_output(messages, maxtoken=70)
                # print("end ask movie name:", time.time() - t1)
                if "i don't know" not in output:
                    try:
                        for tmpoutput in output.split(' && '):
                            tmpoutput = tmpoutput.replace('"', '').strip()
                            context_str += self.retriever.get_movie_context(tmpoutput)
                        print("retrieved movie name:", context_str)
                    except:
                        context_str = ""
                else:
                    context_str = ""
        elif domain in ['music']:
            context_str = self.retriever.get_music_grammy(query)
            print("get_music_grammy:", context_str)
            if context_str is None:
                context_str = ""
            else:
                t1 = time.time()
                filled_template = template_map['output_answer_nofalse'].format(
                    context_str=context_str,
                    query_str=query)
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"
                    },
                    {"role": "user", "content": filled_template},
                ]
                output = self.llama3_output(messages, maxtoken=70)
                print("end music:", output, time.time() - t1)
                if "i don't know" not in output and "invalid" not in output:
                    return output, context_str
                context_str = ""
        elif domain in ['finance']:
            if 'share' in query or 'pe' in query or 'eps' in query or 'ratio' in query or 'capitalization' in query or 'earnings' in query or 'market' in query:
                context_str = ""
                t1 = time.time()
                filled_template = template_map['ask_name_finance'].format(query_str=query)
                messages = [
                    {
                        "role": "system",
                        "content": f" You will be asked a lot of questions, but you don't need to answer them, just point out the specific stock ticker or company name involved."
                    },
                    {
                        "role": "user", 
                        "content": filled_template
                    },
                ]
                output = self.llama3_output(messages, maxtoken=70)
                print("end ask name:", output, time.time() - t1)
                if "i don't know" not in output and 'none' not in output:
                    try:
                        for tmpoutput in output.split(' && '):
                            tmpoutput = tmpoutput.replace('"', '').strip()
                            context_str += self.retriever.get_finance_context(tmpoutput)
                        print("retrieved name:", context_str)
                        t1 = time.time()
                        filled_template = template_map['output_answer_nofalse'].format(
                            context_str=context_str,
                            query_str=query)
                        messages = [
                            {
                                "role": "system",
                                "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 30 words or less. If you are not sure about the query, answer i don't know. There is no need to explain the reasoning behind your answers. Now is {query_time}"
                            },
                            {
                                "role": "user", 
                                "content": filled_template
                            },
                        ]
                        output = self.llama3_output(messages, maxtoken=70)
                        # print("end finance:", time.time() - t1)
                        if "i don't know" not in output and "invalid" not in output:
                            return output, context_str
                        context_str = ""
                    except:
                        context_str = ""
                else:
                    context_str = ""
        
        return "", context_str

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        self.all_st = time.time()
        self.all_time = 500
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]
        answers = []

        # Retrieve top matches for the whole batch
        for _idx in range(len(batch_interaction_ids)):
            query = queries[_idx]
            query_time = query_times[_idx]
            search_results = batch_search_results[_idx]
            
            domain = self.llama3_domain(query)
            print("judge domain", domain)
            context_str = ""
            output, context_str = self.process_task1(domain, query, query_time)
            
            if output != "":
                answers.append(output)
                continue
            
            t1 = time.time()
            if self.retriever.init_retriever(search_results, query=query, task3=(self.Task == 3)):
                search_empty = 0
            else:
                search_empty = 1
            
            # print("build retriever time:", time.time() - t1)
            # print("start query")
            
            t1 = time.time()
            if (search_empty):
                res = [""]
            else:
                res = self.retriever.get_result(query, k=self.k)
            
            for snippet in res[:]:
                context_str += "<DOC>\n" + snippet + "\n</DOC>\n"
            
            context_str = self.tokenizer.encode(
                context_str, max_length=4000, add_special_tokens=False)
            # print('len context_str', len(context_str))
            
            if len(context_str) >= 4000:
                context_str = self.tokenizer.decode(context_str) + "\n</DOC>\n"
            else:
                context_str = self.tokenizer.decode(context_str)
            
            # print("query time:", time.time() - t1)
            filled_template = template_map['output_answer_nofalse'].format(context_str=context_str, query_str=query)
            
            messages = [
                {"role": "system",
                "content": f"You are a helpful and honest assistant. Please, respond concisely and truthfully in 70 words or less. Now is {query_time}"},
                {"role": "user", "content": filled_template},
            ]

            output = self.llama3_output(messages)

            if "i don't know" not in output and output not in ['i', "i don't"]:
                answers.append(output)
            else:
                answers.append("i don't know")

        return answers
    
    def format_chat(self, messages):
        # print(f"Messages: {messages}")

        if self.is_server:
            # there is no need to wrap the messages into chat when using the server
            # because we use the chat API: chat.completions.create
            formatted_prompts = messages
        else:
            formatted_prompts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return formatted_prompts
