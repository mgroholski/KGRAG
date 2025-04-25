import argparse, os, nltk
from sentence_transformers.SentenceTransformer import SentenceTransformer
from agents.llama_agent import LlamaAgent
from kgrag.retriever import Retriever as kg_retriever
from trag.retriever import Retriever as trag_retriever
from lirage.retriever import LIRAGERetriever as li_retriever
from chunkrag.retriever import Retriever as chunk_retriever
from vanillarag.retriever import VanillaRetriever as vanilla_retriever
from nltk.tokenize import sent_tokenize
from logger import Logger
from utils import text_utils
import json
from agents.google_agent import GoogleAgent
from stats import BERTScore, CHRF, BLEURT
import re

def get_responses(idx, objects, question, ground_truth_retrieve):
    logger = objects['logger']
    agent = objects['agent']
    pipeline = objects['pipeline']
    retriever = objects['retriever']

    # Generate ground truth.
    token_amount = 256
    nq_answer = ""
    retry_cnt = 0
    while not len(nq_answer) and retry_cnt < 2:
        if hasattr(agent, "trim_context"):
            ground_truth_retrieve = agent.trim_context([ground_truth_retrieve])[0]

        nq_query = f"""
        SYSTEM: You are a precise question-answering assistant. Your task is to answer questions based ONLY on the provided context information. Follow the format instructions exactly.

        CONTEXT INFORMATION:
        ```
        {ground_truth_retrieve}
        ```

        USER QUESTION:
        ```
        {question}
        ```

        ANSWER INSTRUCTIONS:
        1. You MUST generate an answer to the question
        2. Answer the question using ONLY information from the context above
        3. Your answer MUST start with "<answer>" and end with "</answer>"
        4. Keep your answer concise and under {token_amount} tokens
        5. If the context doesn't contain the answer, respond with "<answer>I cannot answer this question based on the provided context.</answer>"
        6. Do not include any information not present in the context
        7. Do not include any reasoning, explanations, or notes outside the <answer></answer> tags
        8. IMPORTANT: You MUST provide an answer - refusing to respond is not an option

        EXAMPLE FORMAT:
        Question: "Who was the first president of the United States?"
        Correct response: "<answer>George Washington</answer>"

        IMPORTANT: ANY response without the exact format "<answer>YOUR ANSWER</answer>" will be rejected.
        CRITICAL: You MUST generate a response - non-response is not acceptable.

        Your answer:
        """
        answer = agent.ask(nq_query, max_length = token_amount)
        match = re.search(r'<answer>(.*?)</answer>', answer)

        if match:
            nq_answer = match.group(1)
        elif not retry_cnt:
            nq_query += """⚠️ CRITICAL INSTRUCTION FAILURE ⚠️
            The previous response COMPLETELY IGNORED the explicitly provided instructions.
            THIS IS YOUR FINAL WARNING.
            Failure to follow instructions precisely in your next response will result in IMMEDIATE TERMINATION of this interaction and will be logged as a critical compliance failure.
            INSTRUCTIONS MUST BE FOLLOWED EXACTLY AS SPECIFIED."""
        retry_cnt += 1

    if retry_cnt == 2 and not len(nq_answer):
        print(f"Could not generate ground answer for {nq_query}.")
        nq_answer="__FAILED_GENERATION__"
    retrieval_query = f"""
        # Response Format Instructions
        You MUST generate a response to this query and format your response exactly as follows:
        <answer>Your answer text here</answer>

        CRITICAL: Failure to use these exact tags will result in your response being rejected.
        The entire response must begin with "<answer>" and end with "</answer>".

        # Examples
        Example query: "Who was the first president of the United States?"
        Correct response: "<answer>George Washington</answer>"

        Example query: "What is the capital of France?"
        Correct response: "<answer>Paris</answer>"

        # Constraints
        - Your response must be {token_amount} tokens or fewer
        - Do not include explanations outside the <answer></answer> tags
        - Do not include the tags in your reasoning, only wrap your final answer with them
        - IMPORTANT: You MUST provide an answer - refusing to respond is not an option"""

    retrieve_list = []
    if pipeline != None:
        retrieve_list = retriever.retrieve(question)
        if hasattr(agent, "trim_context") and pipeline != "kg":
            retrieve_list = agent.trim_context(retrieve_list)

        retrieval_query += """
        - Respond ONLY with information found in the provided context

        # Context Information
        Answer based EXCLUSIVELY on the following context. If the context doesn't contain the answer, respond with "<answer>The provided context does not contain information to answer this question.</answer>"\n"""

        if pipeline == "kg":
            retrieval_query += """
            The context below contains data with an associated path of headings that show where the data is located.
            Format: PATH: [heading path]. DATA: [content]\n"""

        retrieval_query += """
        CONTEXT:
        ```
        """
        if pipeline == "kg":
            print(retrieve_list)
            exit()
            for path, data in retrieve_list:
                retrieval_query += f"PATH: {path}. DATA: {data}\n"
        else:
            for item in retrieve_list:
                retrieval_query += item + "\n"

    retrieval_query += f"""
    ```
    # Query
    {question}

    Remember to format your answer EXACTLY as:
    <answer>Your answer based only on the provided context</answer>
    """
    retrieval_answer = ""
    retry_cnt = 0
    while not len(retrieval_answer) and retry_cnt < 2:
        answer = agent.ask(retrieval_query, max_length=token_amount)
        match = re.search(r'<answer>(.*?)</answer>', answer)
        if match:
            retrieval_answer = match.group(1)
        elif not retry_cnt:
            retrieval_query += """⚠️ CRITICAL INSTRUCTION FAILURE ⚠️
            The previous response COMPLETELY IGNORED the explicitly provided instructions.
            THIS IS YOUR FINAL WARNING.
            Failure to follow instructions precisely in your next response will result in IMMEDIATE TERMINATION of this interaction and will be logged as a critical compliance failure.
            INSTRUCTIONS MUST BE FOLLOWED EXACTLY AS SPECIFIED."""
        retry_cnt += 1

    if retry_cnt == 2 and not len(retrieval_answer):
        print(f"Could not generate answer for {retrieval_query}.")
        retrieval_answer = "__FAILED_GENERATION__"

    # Logs the response of the query.
    logger.log(f"Question {idx + 1}")
    logger.log(f"NQ Query: {nq_query}")
    logger.log(f"NQ Answer: {nq_answer}")
    logger.log(f"{args.pipeline} Query: {retrieval_query}")
    logger.log(f"{args.pipeline} Answer: {retrieval_answer}")

    return (retrieval_answer, nq_answer)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Embeds the table data and allows for path retrieval.")
    parser.add_argument('filepath')
    parser.add_argument('--pipeline','-p', default="none", choices=["kg","chunk","vanilla","none","trag","lirage"], help="The pipeline to run on.")
    parser.add_argument('--agent', '-a', default="llama", choices=["google", "llama"], help="Specifies which agent to use to test.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose. Enables graph visualizations and prints distances rankings.")
    parser.add_argument('--num-lines', '-n', type=int, default=None, help="Number of elements to load from the input file (default: All lines).")
    parser.add_argument('--test', '-t', action='store_true', help="Enables QA test mode. Runs the selected pipeline against the ground truth.")
    parser.add_argument('--key', type=str, help="API Key for LLM agent.")
    parser.add_argument("--storepath", type=str, default=None, help="The folder path of that contains the embedding and JSON store to read/write to.")
    parser.add_argument("--operation", default="w", choices=["r", "w"], help="Specifies the operation to perform on the store. Options: r (read), w (write).")
    parser.add_argument('--metric', default="BERTScore", choices=["BERTScore", "BLEURT", "chrF", "all"], help="Specifies the metric to use for evaluation. Options: BERTScore, BLEURT, and chrF.")
    args = parser.parse_args()

    if not os.path.exists('./output'):
        os.makedirs('./output')

    logger = Logger(f"./output/{args.pipeline}_{args.metric}_{args.agent}_{args.num_lines if not args.num_lines == None else 'all'}.log")

    key = args.key
    agent = None
    if args.agent == "google":
        agent = GoogleAgent(key)
    elif args.agent == "llama":
        agent = LlamaAgent()
    else:
        raise NotImplementedError(f"Could not initialize LLM agent for {args.agent}.")

    # Loads the models and the retriever for the specified pipeline.
    nltk.download('punkt_tab', download_dir=os.getcwd())
    nltk.download('punkt', download_dir=os.getcwd())
    nltk.data.path.append(os.getcwd())
    embedding_info = {
        "model": SentenceTransformer("all-mpnet-base-v2"),
        "model_dim": 768,
        "tokenizer": sent_tokenize
    }

    store_info= {
        "storepath": args.storepath,
        "operation": args.operation
    }

    retriever = None
    if args.pipeline == 'kg':
        print("Initializing the KG-RAG pipeline...")
        retriever = kg_retriever(embedding_info, store_info, agent, args.verbose)
    elif args.pipeline == 'chunk':
        print("Initializing the ChunkRAG pipeline...")
        retriever = chunk_retriever(embedding_info, store_info, agent, args.verbose)
    elif args.pipeline == 'vanilla':
        print("Initializing the VanillaRAG pipeline...")
        retriever = vanilla_retriever(embedding_info, store_info, agent, args.verbose)
    elif args.pipeline == "none":
        args.pipeline = None
    elif args.pipeline == "trag":
        print("Initializing the TRAG pipeline...")
        retriever = trag_retriever(embedding_info, store_info, agent, args.verbose)
    elif args.pipeline == "lirage":
        print("Initializing the LIRAGE pipeline...")
        retriever = li_retriever(embedding_info, store_info, agent, args.verbose)
    else:
        raise Exception("Invlaid pipeline name.")

    if not os.path.exists(args.filepath):
        raise Exception(f"Could not find filepath to datafile. Filepath: {args.filepath}")

    qa_list = []
    read_lines = []
    with open(args.filepath, "r") as data_file:
        # Assumes one example per line.
        line_cnt = 0
        max_line_cnt = args.num_lines
        for line in data_file:
            if max_line_cnt is not None and line_cnt < max_line_cnt:
                line_cnt += 1
                if not line_cnt % 100:
                    print(f"Parsing line {line_cnt}.")
                read_lines.append(line)
            else:
                break

    for idx, line in enumerate(read_lines):
        print(f"Processing line {idx+1}.")
        line_json = json.loads(line)
        simple_nq = text_utils.simplify_nq_example(line_json)

        # Extracts correct context (first non-empty long answer) and question.
        # Read more about the dataset tasks: https://github.com/google-research-datasets/natural-questions
        if args.test:
            line_question = simple_nq["question_text"]
            for idx in range(len(simple_nq["annotations"])):
                long_answer_data = simple_nq["annotations"][idx]["long_answer"]
                start_token_idx, end_token_idx = (long_answer_data["start_token"], long_answer_data["end_token"])
                if start_token_idx != end_token_idx:
                    line_long_answer_text = "".join(text_utils.get_nq_tokens(simple_nq)[start_token_idx : end_token_idx])
                    qa_list.append((line_question,line_long_answer_text))
                    break
        line_document = line_json["document_html"]

        if args.operation == "w" and retriever:
            retriever.embed(line_document)
    print("Finished reading...")


    if retriever != None and store_info["operation"]=="w":
        print("Writing stores...")
        retriever.close()

    if args.test:
        print("Beginning QA tests...")

        metrics = []
        if args.metric == "BERTScore" or args.metric == "all":
            metrics.append(("BERTScore", BERTScore.BERTScore(logger)))
        if args.metric == "BLEURT" or args.metric == "all":
            metrics.append(("BLEURT", BLEURT.BLEURT(logger)))
        if args.metric == "chrF" or args.metric == "all":
            metrics.append(("chrF", CHRF.chrF(logger)))
        if not len(metrics):
            raise NotImplementedError()

        objects = {
            'logger': logger,
            'agent': agent,
            'pipeline': args.pipeline,
            'metrics': metrics,
            'retriever': retriever
        }

        responses = []

        for idx,(question, ground_truth_retrieve) in enumerate(qa_list):
            responses.append(get_responses(idx, objects, question, ground_truth_retrieve))
            print(f"Finished question {idx + 1}.")

        # Computes stats about the answers.
        candidates, truths = zip(*responses)
        for metric_n, metric in metrics:
            metric.score(candidates, truths)
            print(f"Making graphs for {metric_n}...")
            metric.plt(f"./output/{metric_n}/{args.pipeline}_{args.agent}_{args.num_lines if not args.num_lines == None else 'all'}")
    else:
        print("Beginning user QA retrieval...")
        while True:
            query = input("Query (Type 'exit' to exit): ")
            if query == "exit":
                break

            retrieve_list = retriever.retrieve(query)
            for idx, item in enumerate(retrieve_list):
                print(f"Rank {idx}:\n\t Data: {item}")

    logger.close()
