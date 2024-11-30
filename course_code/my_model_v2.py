import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
import json

# ATTENTION: pip install bz2, json, langchain, langchain_community
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from transformers import  GenerationConfig
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_core.documents import Document



#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class Retriever:
    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.vector_store = Chroma(embedding_function = self.embeddings)
        self.docstore = InMemoryStore()
        self.retriever = ParentDocumentRetriever(vectorstore = self.vector_store,
                                                  docstore = self.docstore, 
                                                  child_splitter=self.child_splitter, 
                                                  parent_splitter = self.parent_splitter)
    def clean_search_results(self, search_results):
        """
        search_results: List[Dict]
        """
        res = []
        for html_page_json in search_results:
            soup = BeautifulSoup(json.dumps(html_page_json["page_result"]), "html.parser")
            # clean out CSS/style elements
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text(separator=' ')
            # clean lines
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = ' '.join(chunk for chunk in chunks if chunk)
            # create document for each HTML text
            res.append(Document(page_content=text, metadata={}))
        self.retriever.add_documents(res)

    # return list string of retrieved docs
    def get_documents(self, query, search_results, k):
        """
        query: string 
        search_results: List[Dict] -> list holds 5 search query results
        k: int -> gets top k results
        """
        # clean out search results
        search_results = self.clean_search_results(search_results)
        # get relevant documents based on query embeddings + parent/child embeds lez go
        retrieved_docs = self.retriever.get_relevant_documents(query)
        if k > len(retrieved_docs):
            k = len(retrieved_docs) - 1
        return [x.page_content for x in retrieved_docs[:k+1]]
    

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None, k=4):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.retriever = Retriever()
        self.k = k

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
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
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings
    
    def llama3_domain(self, query):
        messages = [
            {"role": "system", "content": f"You are an assistant expert in movie, sports, finance and music fields."},
            {"role": "user",
             "content": "Please judge which category the query belongs to, without answering the query. you can only and must output one word in (movie, sports, finance, music) If the question doesn't belong to movie, sports,finance, music, please answer other. \n Query:" + query + '\n Category:'},
        ]
        domain, _, _ = self.llam3_output(messages, maxtoken=3, disable_adapter=True)
        for key in ['finance', 'music', 'sports', 'movie']:
            if key in domain:
                return key
        return 'open'
    
    def llam3_output(self, messages, maxtoken=75, disable_adapter=False):
        self.m.eval()
        if time.time() - self.all_st >= self.all_time:
            return "i don't know", 0, 0
        with torch.no_grad():
            t1 = time.time()
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.m.device)
            print('input_ids shape', input_ids.shape)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generation_config = GenerationConfig(
                max_new_tokens=maxtoken, do_sample=False,
                max_time=32 - (time.time() - self.t_s), eos_token_id=terminators)
            if disable_adapter:
                with self.m.disable_adapter():
                    outputs = self.m.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        eos_token_id=terminators,
                        return_dict_in_generate=True,
                        output_scores=False)
            else:
                outputs = self.m.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    eos_token_id=terminators,
                    return_dict_in_generate=True,
                    output_scores=False)

            output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).lower().split("assistant")[
                -1].strip()
            print("end gen:", time.time() - t1)
            print("output:")
            print(output)
        return output, 0, 0

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
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Retrieve top matches for the whole batch
        # ATTENTION: CHANGES HERE
        #
        #
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            search_results = batch_search_results[_idx]
            # use chroma vector DB to get automatic top k results for eqch query
            retrieval_results = self.retriever.get_documents(query, search_results, self.k)
            # ATTENTION: retrieval results is a list of strings, idk if this is right format
            batch_retrieval_results.append(retrieval_results)
        # 
        # 
        # 
        
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message 
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts
