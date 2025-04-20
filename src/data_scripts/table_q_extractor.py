import json
import argparse
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from utils import text_utils

# Data Download Link: https://ai.google.com/research/NaturalQuestions/download

def process_line(line):
    """Process a single line and return it if it contains a table, None otherwise."""
    try:
        line_json = json.loads(line)
        simple_nq = text_utils.simplify_nq_example(line_json)
        tokens = text_utils.get_nq_tokens(simple_nq)

        for annotation in simple_nq["annotations"]:
            long_answer = annotation["long_answer"]
            start_token, end_token = long_answer["start_token"], long_answer["end_token"]
            answer_text = " ".join(tokens[start_token:end_token])
            if "<Table>" in answer_text:
                return line
        return None
    except Exception as e:
        print(f"Error processing line: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extracts tables from html.")
    parser.add_argument('filepath')
    parser.add_argument('write_filepath')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(),
                        help='Number of worker processes to use')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Chunk size for multiprocessing')
    args = parser.parse_args()

    # Count total lines for progress bar
    total_lines = sum(1 for _ in open(args.filepath, 'r'))

    # Create a partial function with text_utils already provided
    process_func = partial(process_line)

    write_operations = 0
    with open(args.filepath, "r") as file, \
         open(os.path.expanduser(f"{args.write_filepath}"), "w") as write_file:
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, file, chunksize=args.chunk_size),
                total=total_lines,
                desc="Processing lines"
            ))

            for result in results:
                if result is not None:
                    write_operations += 1
                    write_file.write(result)

    print("Write Operations: ", write_operations)
    print("Lines: ", total_lines)
if __name__ == "__main__":
    main()
