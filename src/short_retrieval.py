import argparse
import text_utils
import json
import graph_utils
from visualizations.graph_visualizations import visualize_graph_topological, visualize_graph_with_cycle_detection
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import os
import nltk
from numpy import dot
from numpy.linalg import norm
import heapq
import re

def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')

    nltk.download('punkt_tab', download_dir=os.getcwd())
    nltk.data.path.append(os.getcwd())
    tokenizer = sent_tokenize

    return {
        "model": model,
        "tokenizer": tokenizer
    }

def main(filepath: str, visualize: bool, num_lines: int, test_mode: bool):
    model_dict = load_model()
    model, tokenizer = model_dict["model"], model_dict["tokenizer"]

    path_texts_and_data = []
    path_embeddings = []

    test_qa = []
    with open(filepath, "r") as read_file:
        line_cnt = 0
        for line in read_file:
            if line_cnt < num_lines:
                line_cnt += 1

                line_json = json.loads(line)
                simple_nq = text_utils.simplify_nq_example(line_json)
                simple_text = simple_nq["document_text"]

                if test_mode:
                    line_question = simple_nq["question_text"]
                    line_html_tokens = text_utils.get_nq_tokens(simple_nq)
                    answers = []

                    for annotation in simple_nq["annotations"]:
                        long_answer_data = annotation["long_answer"]
                        long_answer = line_html_tokens[long_answer_data["start_token"]:long_answer_data["end_token"]]
                        answers.append(long_answer)

                    test_qa.append({
                        "question": line_question,
                        "answers": answers
                    })

                page_graph = graph_utils.extract_relationship_graphs(simple_text)
                if visualize:
                    try:
                        visualize_graph_topological(page_graph)
                    except Exception:
                        visualize_graph_with_cycle_detection(page_graph)
                    exit()

                # Paths from source to leaf
                stack = [[page_graph]]
                while stack:
                    path = stack.pop()
                    u = path[-1]

                    if not len(u.adj):
                        # If leaf node, append path text
                        path2text = ", ".join([re.sub(r'[\.|\?|!]', '', v.value) for v in path])
                        path_texts_and_data.append((path2text, u.data))
                        path_tokens = tokenizer(path2text, language="english")
                        path_embedding = model.encode(path_tokens)
                        if path_embedding.shape[0] > 1:
                            raise NotImplementedError(f"Path embedding with multiple embedding vectors for \"{path2text}\"\n\nTokens: {path_tokens}\n\nShape:{path_embedding.shape}.")
                        path_embeddings.append(path_embedding[0])
                    else:
                        for v in u.adj:
                            stack.append(path + [v])

        print("Writing embeddings...")
        with open("data/50_embeddings_data.json", "w") as embeddings_file:
            json_arr = []

            for (text, data), embedding in zip(path_texts_and_data, path_embeddings):
                json_obj = {
                    "text": text,
                    "data": data,
                    "embedding": repr(embedding.tolist())
                }

                json_arr.append(json_obj)
            embeddings_file.write(json.dumps(json_arr))
        print("Finished writing embeddings.")

    print("Starting retrieval...")

    if test_mode:
        '''
            For each question,
                1. +1 if long answer is in top_k
                2. Create accuracy percentage for question
            Take average of all percentages.
        '''

        pass
    else:
        while True:
            query = input("Query (Type 'exit' to exit): ")
            if query == "exit":
                break

            query_tokens = tokenizer(query, language="english")
            query_embeddings = model.encode(query_tokens)

            top_k = 20
            min_heap = []
            for idx, embedding in enumerate(path_embeddings):
                score = 0

                for q_embedding in query_embeddings:
                    score += (dot(embedding, q_embedding)) / (norm(embedding) * norm(q_embedding))

                score /= len(query_tokens)
                heapq.heappush(min_heap, (score, path_texts_and_data[idx]))

                while len(min_heap) > top_k:
                    heapq.heappop(min_heap)

            stack = []
            while min_heap:
                stack.append(heapq.heappop(min_heap))

            ranking = 1
            while stack:
                score, (path_text, path_data) = stack.pop()
                print(f"Text Rank {ranking}:\n\tText: \"{path_text}\"\n\tData:{path_data}\n\tScore:{score}\n")
                ranking += 1


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Embeds the table data and allows for path retrieval.")
    parser.add_argument('filepath')
    parser.add_argument('--visualize', '-v', action='store_true', help="Enable graph visualization")
    parser.add_argument('--num-lines', '-n', type=int, default=50, help="Number of elements to load from the input file (default: 50)")
    parser.add_argument('--test', '-t', action='store_true', help="Enables test mode. Returns statistics.")

    args = parser.parse_args()
    main(args.filepath, args.visualize, args.num_lines, args.test)
