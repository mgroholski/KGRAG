# KG-RAG

## Dataset
We utilize NQ Table which is a subset of NQ. The NQ data can be downloaded at [Google AI](https://ai.google.com/research/NaturalQuestions/download). We filter this table to utilize questions that derive their answers
from tables (`src/data_scripts/table_q_extractor.py`).

## Usage

To run the script, download the required libraries from `requirements.txt` use the following command:

```bash
python3 src/main.py <filepath> [options]
```

### Command Line Arguments

- `filepath`: Path to the input file containing the dataset (required)
- `--pipeline`, `-p`: Specifies which pipeline to run. Options: `kg` (default), `chunk`, `vanilla`
- `--agent`, `-a`: Specifies which agent to use. Options: `google` (default)
- `--verbose`, `-v`: Enables verbose mode with graph visualizations and distance rankings
- `--num-lines`, `-n`: Number of lines to load from the input file (default: all lines)
- `--test`, `-t`: Enables QA test mode to compare against ground truth
- `--key`: API Key for the LLM agent
- `--storepath`: Folder path to read/write embedding and JSON store
- `--operation`: Specifies the operation to perform on the store. Options: `r` (read), `w` (write, default)
- `--metric`: Metric to use for evaluation. Options: `BERTScore` (default), `BLEURT`, `chrF`

### Examples

Run KG-RAG pipeline on 100 examples using Google agent with test mode:
```bash
python src/main.py data/nq_examples.jsonl --pipeline kg --agent google --num-lines 100 --test --key YOUR_API_KEY
```

Run 250 questions using ChunkRAG with the BERTScore metric and reading from previously stored folder.

```bash
python3 src/main.py data/filtered_table_short_answer.jsonl -p chunk -n 250 -t --key API_KEY --storepath stores/chunk_250 --metric BERTScore --operation r
```
