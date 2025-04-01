import argparse
import BERTScore
import text_utils
import json
import graph_utils
from visualizations.graph_visualizations import visualize_graph_topological, visualize_graph_with_cycle_detection
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import os
import nltk
import re
from graph_utils import NodeTagType
from retriever import Retriever
from google_agent import GoogleAgent

def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')

    nltk.download('punkt_tab', download_dir=os.getcwd())
    nltk.data.path.append(os.getcwd())
    tokenizer = sent_tokenize

    return {
        "model": model,
        "tokenizer": tokenizer
    }

def main(filepath: str, visualize: bool, num_lines: int, test_mode: bool, key: str):
    model_dict = load_model()
    model, tokenizer = model_dict["model"], model_dict["tokenizer"]

    retriever = Retriever(model, tokenizer)
    test_qa = []
    with open(filepath, "r") as read_file:
        line_cnt = 0
        for line in read_file:
            if line_cnt < num_lines:
                print(f"Parsing line {line_cnt}.")
                line_cnt += 1

                line_json = json.loads(line)
                simple_nq = text_utils.simplify_nq_example(line_json)
                simple_text = simple_nq["document_text"]

                if test_mode:
                    line_question = simple_nq["question_text"]
                    line_html_tokens = text_utils.get_nq_tokens(simple_nq)
                    long_answers = []
                    short_answers = []

                    for annotation in simple_nq["annotations"]:
                        long_answer_data = annotation["long_answer"]
                        long_answers.append("".join(line_html_tokens[long_answer_data["start_token"]:long_answer_data["end_token"]]))

                        short_answers_data = annotation["short_answers"]
                        short_answers_set = set()
                        for short_answer_data in short_answers_data:
                            short_answers_set.add("".join(line_html_tokens[short_answer_data["start_token"]:short_answer_data["end_token"]]))
                        short_answers.append(frozenset(short_answers_set))

                    test_qa.append({
                        "question": line_question,
                        "answer": long_answers,
                        "truth": short_answers
                    })

                page_graph = graph_utils.extract_relationship_graphs(simple_text)
                if visualize:
                    try:
                        visualize_graph_topological(page_graph)
                    except Exception as e:
                        print(e)
                        visualize_graph_with_cycle_detection(page_graph)
                        exit()

                # Paths from source to leaf
                path_texts_and_data = []
                stack = [[page_graph]]
                while stack:
                    path = stack.pop()
                    u = path[-1]

                    if not len(u.adj) and u.type == NodeTagType.TH:
                        # If leaf node, append path text
                        path2text = ", ".join([re.sub(r'[\.|\?|!]', '', v.value) for v in path])
                        path_texts_and_data.append((path2text, u.data))
                    else:
                        for v in u.adj:
                            stack.append(path + [v])

                corpus, data = zip(*path_texts_and_data)
                retriever.embed(corpus, data)
            else:
                break

    print("Starting retrieval...")
    if test_mode:
        if not key:
            print("Cannot query LLM without key.")
            exit()

        agent = GoogleAgent(key)
        similarity_scores = []
        for obj in test_qa:
            question, q_long_answer = obj['question'], obj['answer'][0]

            q_query = f"""{q_long_answer}

            Answer this question: {question}"""
            q_query_answer = agent.ask(q_query, max_length = 500)
            retrieve_list = retriever.retrieve(question)
            retrieve_query = "This is data retrieved from Wikipedia:\n"
            for _, (retrieve_path, retrieve_data) in retrieve_list:
                retrieve_query += f"""
                Retrieve Path: \"{retrieve_path}\",
                Retrieve Data: {retrieve_data},

                """
            retrieve_query += f"Answer this question: {question}"
            retrieve_query_answer = agent.ask(retrieve_query, max_length = 500)

            print(f"Q query: {q_query}")
            print(f"Q answer: {q_query_answer}")
            print(f"Retrieve query: {retrieve_query}")
            print(f"Retrieve answer: {retrieve_query_answer}")

            similarity_scores.append((BERTScore.calculate_bert_score([retrieve_query_answer], [q_query_answer]), question))
        # After all questions are processed, generate the plots
        print("Generating BERTScore plots...")
        import matplotlib.pyplot as plt

        # Extract data from similarity_scores
        precision_scores = [score[0]['bert_precision'] for score in similarity_scores]
        recall_scores = [score[0]['bert_recall'] for score in similarity_scores]
        f1_scores = [score[0]['bert_f1'] for score in similarity_scores]

        # Generate question labels (q1, q2, q3, ...)
        questions = [f"q{i+1}" for i in range(len(similarity_scores))]

        # Create plots for every 50 questions
        for i in range(0, len(questions), 50):
            plot_start = i
            plot_end = min(i + 50, len(questions))

            # Get the relevant slice of data for this plot
            plot_questions = questions[plot_start:plot_end]
            plot_precision = precision_scores[plot_start:plot_end]
            plot_recall = recall_scores[plot_start:plot_end]
            plot_f1 = f1_scores[plot_start:plot_end]

            # Create a figure with three subplots for precision, recall, and F1
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

            # Plot precision
            ax1.bar(plot_questions, plot_precision, color='blue')
            ax1.set_title(f'BERTScore Precision (Questions {plot_start+1}-{plot_end})')
            ax1.set_xlabel('Questions')
            ax1.set_ylabel('Precision Score')
            ax1.set_ylim(0, 1)

            # Plot recall
            ax2.bar(plot_questions, plot_recall, color='green')
            ax2.set_title(f'BERTScore Recall (Questions {plot_start+1}-{plot_end})')
            ax2.set_xlabel('Questions')
            ax2.set_ylabel('Recall Score')
            ax2.set_ylim(0, 1)

            # Plot F1
            ax3.bar(plot_questions, plot_f1, color='red')
            ax3.set_title(f'BERTScore F1 (Questions {plot_start+1}-{plot_end})')
            ax3.set_xlabel('Questions')
            ax3.set_ylabel('F1 Score')
            ax3.set_ylim(0, 1)

            # Rotate x-axis labels if there are many questions
            for ax in [ax1, ax2, ax3]:
                if len(plot_questions) > 20:
                    plt.setp(ax.get_xticklabels(), rotation=90)

            plt.tight_layout()
            plt.savefig(f'./output/bertscore_questions_{plot_start+1}_to_{plot_end}.png')
            print(f"Created plot for questions {plot_start+1} to {plot_end}")
            plt.close()

            # Print overall statistics
            avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
            avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

            print("Similarity Scores: ", similarity_scores)
            print(f"\nOverall BERTScore Statistics:")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            print(f"Average F1 Score: {avg_f1:.4f}")

            # After calculating avg_precision, avg_recall, avg_f1
            # Create a summary table for total averages
            print("\nCreating summary table for overall averages...")
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)

            # Table data
            table_data = [
                ['Metric', 'Average Value'],
                ['Precision', f'{avg_precision:.4f}'],
                ['Recall', f'{avg_recall:.4f}'],
                ['F1 Score', f'{avg_f1:.4f}']
            ]

            # Create table
            table = plt.table(cellText=table_data,
                              colWidths=[0.3, 0.3],
                              loc='center',
                              cellLoc='center')

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)

            # Hide axis
            ax.axis('off')

            # Add a title
            plt.title('BERTScore Summary Statistics', pad=20)

            # Save the table
            plt.savefig('./output/bertscore_summary_table.png', bbox_inches='tight')
            plt.close()
            print("Created summary table image")
    else:
        while True:
            query = input("Query (Type 'exit' to exit): ")
            if query == "exit":
                break

            retrieve_list = retriever.retrieve(query)

            ranking = 1
            while retrieve_list:
                score, (path_text, path_data) = retrieve_list.pop(0)
                print(f"Text Rank {ranking}:\n\tText: \"{path_text}\"\n\tData:{path_data}\n\tScore:{score}\n")
                ranking += 1


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Embeds the table data and allows for path retrieval.")
    parser.add_argument('filepath')
    parser.add_argument('--visualize', '-v', action='store_true', help="Enable graph visualization")
    parser.add_argument('--num-lines', '-n', type=int, default=50, help="Number of elements to load from the input file (default: 50)")
    parser.add_argument('--test', '-t', action='store_true', help="Enables test mode. Returns statistics.")
    parser.add_argument('--key', type=str, help="API Key for LLM agent.")

    args = parser.parse_args()
    main(args.filepath, args.visualize, args.num_lines, args.test, args.key)
