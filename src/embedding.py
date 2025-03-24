import argparse
import text_utils
import json
import graph_utils
from visualizations.graph_visualizations import visualize_graph_topological

def main(filepath: str):
    with open(filepath, "r") as read_file:
        i = 0
        for line in read_file:
            line_json = json.loads(line)
            simple_nq = text_utils.simplify_nq_example(line_json)
            simple_text = simple_nq["document_text"]
            page_graph = graph_utils.extract_relationship_graphs(simple_text)

            if i < 5:
                visualize_graph_topological(page_graph)
                i += 1
            else:
                exit()


            # TODO: Embed

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Embeds the table data and stores within a vector database.")
    parser.add_argument('filepath')

    args = parser.parse_args()
    main(args.filepath)
