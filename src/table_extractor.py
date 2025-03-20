import json
import argparse
import text_utils
import os

parser = argparse.ArgumentParser(description="Extracts tables from html.")
parser.add_argument('filepath')
args = parser.parse_args()

with open(args.filepath, "r") as file:
    with open(os.path.expanduser(f"data/filtered_{os.path.basename(args.filepath)}"), "w") as write_file:
        for idx, line in enumerate(file):
            line_json = json.loads(line)
            simple_nq = text_utils.simplify_nq_example(line_json)



            exit()
