import json
import argparse
import text_utils
import os

# Data Download Link: https://ai.google.com/research/NaturalQuestions/download
write_operations = 0
lines = 0

parser = argparse.ArgumentParser(description="Extracts tables from html.")
parser.add_argument('filepath')
args = parser.parse_args()

with open(args.filepath, "r") as file:
    with open(os.path.expanduser(f"data/filtered_{os.path.basename(args.filepath)}"), "w") as write_file:
        for idx, line in enumerate(file):
            lines += 1
            line_json = json.loads(line)
            simple_nq = text_utils.simplify_nq_example(line_json)

            tokens = text_utils.get_nq_tokens(simple_nq)
            long_answer_candidates = simple_nq["long_answer_candidates"]

            for candidate in long_answer_candidates:
                start_token, end_token = candidate["start_token"], candidate["end_token"]
                candidate_text = " ".join(tokens[start_token:end_token])
                if "<Table>" in candidate_text:
                    write_operations += 1
                    write_file.write(line)
                    break

print("Write Operations: ", write_operations)
print("Lines: ", lines)
