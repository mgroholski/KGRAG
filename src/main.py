import argparse, os, nltk
from sentence_transformers.SentenceTransformer import SentenceTransformer
from kgrag.retriever import Retriever as kg_retriever
from chunkrag.retriever import Retriever as chunk_retriever
from nltk.tokenize import sent_tokenize
from logger import Logger
from utils import text_utils
import json
from agents.google_agent import GoogleAgent
from stats import BERTScore, CHRF, BLEURT

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Embeds the table data and allows for path retrieval.")
    parser.add_argument('filepath')
    parser.add_argument('--pipeline','-p', default="kg", choices=["kg","chunk","vanilla"], help="The pipeline to run on.")
    parser.add_argument('--agent', '-a', default="google", choices=["google"], help="Specifies which agent to use to test.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose. Enables graph visualizations and prints distances rankings.")
    parser.add_argument('--num-lines', '-n', type=int, default=None, help="Number of elements to load from the input file (default: All lines).")
    parser.add_argument('--test', '-t', action='store_true', help="Enables QA test mode. Runs the selected pipeline against the ground truth.")
    parser.add_argument('--key', type=str, help="API Key for LLM agent.")
    parser.add_argument("--storepath", type=str, default=None, help="The folder path of that contains the embedding and JSON store to read/write to.")
    parser.add_argument("--operation", default="w", choices=["r", "w"], help="Specifies the operation to perform on the store. Options: r (read), w (write).")
    parser.add_argument('--metric', default="BERTScore", choices=["BERTScore", "BLEURT", "chrF"], help="Specifies the metric to use for evaluation. Options: BERTScore, BLEURT, and chrF.")
    args = parser.parse_args()

    if not os.path.exists('./output'):
        os.makedirs('./output')

    logger = Logger(f"./output/{args.pipeline}_{args.metric}_{args.agent}_{args.num_lines if not args.num_lines == None else 'all'}.log")

    key = args.key
    agent = None
    if args.agent == "google":
        agent = GoogleAgent(key)
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
        "storepath": args.storepath
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
    else:
        raise Exception("Invlaid pipeline name.")

    if not os.path.exists(args.filepath):
        raise Exception(f"Could not find filepath to datafile. Filepath: {args.filepath}")


    logger.log("Reading dataset and creating embeddings...")

    qa_list = []
    with open(args.filepath, "r") as data_file:
        # Assumes one example per line.
        line_cnt = 0
        max_line_cnt = args.num_lines
        for line in data_file:
            if max_line_cnt is not None and line_cnt < max_line_cnt:
                line_cnt += 1
                print(f"Parsing line {line_cnt}.")
                if not line_cnt % 100:
                    logger.log(f"Parsing line {line_cnt}.")

                line_json = json.loads(line)
                simple_nq = text_utils.simplify_nq_example(line_json)

                # Extracts correct context (first long answer) and question.
                # Read more about the dataset tasks: https://github.com/google-research-datasets/natural-questions
                if args.test:
                    line_question = simple_nq["question_text"]
                    long_answer_data = simple_nq["annotations"][0]["long_answer"]
                    line_long_answer_text = "".join(text_utils.get_nq_tokens(simple_nq)[long_answer_data["start_token"] : long_answer_data["end_token"]])
                    qa_list.append((line_question,line_long_answer_text))

                line_document = simple_nq["document_text"]
                if args.operation == "w":
                    retriever.embed(line_document)
            else:
                break

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


        responses = []
        for question, ground_truth_retrieve in qa_list:
            # Generate ground truth.
            nq_query = f"""Use only the context to answer the query.
            CONTEXT:
                {ground_truth_retrieve}

            QUERY:
                {question}
            """

            nq_answer = agent.ask(nq_query, max_length = 500)

            # Generate retrieval answer.
            retrieve_list = retriever.retrieve(question)
            retrieval_query = "Use only the context to answer the query. "
            if args.pipeline == "kg":
                retrieval_query += "We will provide data as context with an associated path of headings that lead to where to the data is located.\n"
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
            logger.log(f"NQ Query: {nq_query}")
            logger.log(f"NQ Answer: {nq_answer}")
            logger.log(f"{args.pipeline} Query: {retrieval_query}")
            logger.log(f"{args.pipeline} Answer: {retrieval_answer}")

            responses.append((retrieval_answer, nq_answer))
            if not len(responses) % 10:
                print(f"{len(responses)} responses.")

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
