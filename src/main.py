import argparse, os, nltk
from sentence_transformers.SentenceTransformer import SentenceTransformer
from agents.llama_agent import LlamaAgent
from kgrag.retriever import Retriever as kg_retriever
from chunkrag.retriever import Retriever as chunk_retriever
from nltk.tokenize import sent_tokenize
from logger import Logger
from utils import text_utils
import json
from agents.google_agent import GoogleAgent
from stats import BERTScore, CHRF, BLEURT
import concurrent.futures

def get_responses(idx, objects, question, ground_truth_retrieve):
    logger = objects['logger']
    agent = objects['agent']
    pipeline = objects['pipeline']


    # Generate ground truth.
    nq_query = f"""Use only the context to answer the query.
    CONTEXT:
        {ground_truth_retrieve}

    QUERY:
        {question}
    """

    nq_answer = agent.ask(nq_query, max_length = 500)

    # Generate retrieval answer.
    retrieval_query = ""
    retrieve_list = []
    if pipeline != None:
        retrieve_list = retriever.retrieve(question)
        retrieval_query = "Use only the context to answer the query. "

    if pipeline == "kg":
        retrieval_query += "We will provide data as context with an associated path of headings that lead to where to the data is located.\n"
    if pipeline != None:
        retrieval_query += """
        CONTEXT:\n"""
        if args.pipeline == "kg":
            for path, data in retrieve_list:
                retrieval_query += f"PATH: {path}. DATA: {data}\n"
        else:
            for item in retrieve_list:
                retrieval_query += item + "\n"
    retrieval_query += f"""
    QUERY:
        {question}
    """

    retrieval_answer = agent.ask(retrieval_query, max_length = 500)

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
    parser.add_argument('--pipeline','-p', default="none", choices=["kg","chunk","vanilla","none"], help="The pipeline to run on.")
    parser.add_argument('--agent', '-a', default="llama", choices=["google"], help="Specifies which agent to use to test.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose. Enables graph visualizations and prints distances rankings.")
    parser.add_argument('--num-lines', '-n', type=int, default=None, help="Number of elements to load from the input file (default: All lines).")
    parser.add_argument('--test', '-t', action='store_true', help="Enables QA test mode. Runs the selected pipeline against the ground truth.")
    parser.add_argument('--key', type=str, help="API Key for LLM agent.")
    parser.add_argument("--storepath", type=str, default=None, help="The folder path of that contains the embedding and JSON store to read/write to.")
    parser.add_argument("--operation", default="w", choices=["r", "w"], help="Specifies the operation to perform on the store. Options: r (read), w (write).")
    parser.add_argument('--metric', default="BERTScore", choices=["BERTScore", "BLEURT", "chrF"], help="Specifies the metric to use for evaluation. Options: BERTScore, BLEURT, and chrF.")
    parser.add_argument('--threads', '-th', default=1, type=int, help="Number of threads to use.")
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
        logger.log("Initializing the KG-RAG pipeline...")
        retriever = kg_retriever(embedding_info, store_info, agent, args.verbose)
    elif args.pipeline == 'chunk':
        logger.log("Initializing the ChunkRAG pipeline...")
        retriever = chunk_retriever(embedding_info, store_info, agent, args.verbose)
    elif args.pipeline == 'vanilla':
        logger.log("Initializing the VanillaRAG pipeline...")
        # TODO: Add vanilla rag
        raise NotImplementedError()
    elif args.pipeline == "none":
        args.pipeline = None
    else:
        raise Exception("Invlaid pipeline name.")

    if not os.path.exists(args.filepath):
        raise Exception(f"Could not find filepath to datafile. Filepath: {args.filepath}")


    logger.log("Reading dataset and creating embeddings...")

    qa_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        embed_futures = []
        with open(args.filepath, "r") as data_file:
            # Assumes one example per line.
            line_cnt = 0
            max_line_cnt = args.num_lines
            for line in data_file:
                if max_line_cnt is not None and line_cnt < max_line_cnt:
                    line_cnt += 1
                    if not line_cnt % 100:
                        logger.log(f"Parsing line {line_cnt}.")

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
                    if args.operation == "w":
                        future = executor.submit(retriever.embed, line_document)
                        embed_futures.append(future)
                else:
                    break

            for future in embed_futures:
                future.result()

    if args.test:
        logger.log("Beginning QA tests...")

        metric = None
        if args.metric == "BERTScore":
            metric = BERTScore.BERTScore(logger)
        elif args.metric == "BLEURT":
            metric = BLEURT.BLEURT(logger)
        elif args.metric == "chrF":
            metric = CHRF.chrF(logger)
        else:
            raise NotImplementedError()

        objects = {
            'logger': logger,
            'agent': agent,
            'pipeline': args.pipeline,
            'metric': metric
        }

        responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            retrieve_futures = []
            for idx,(question, ground_truth_retrieve) in enumerate(qa_list):
                future = executor.submit(get_responses, idx, objects, question, ground_truth_retrieve)
                retrieve_futures.append(future)

            for future in retrieve_futures:
                result = future.result()
                responses.append(result)

        # Computes stats about the answers.
        candidates, truths = zip(*responses)
        metric.score(candidates, truths)
        print("Making graphs...")
        metric.plt(f"./output/{args.metric}/{args.pipeline}_{args.agent}_{args.num_lines if not args.num_lines == None else 'all'}")
    else:
        logger.log("Beginning user QA retrieval...")
        while True:
            query = input("Query (Type 'exit' to exit): ")
            if query == "exit":
                break

            retrieve_list = retriever.retrieve(query)
            for idx, item in enumerate(retrieve_list):
                print(f"Rank {idx}:\n\t Data: {item}")

    logger.close()
    retriever.close()
