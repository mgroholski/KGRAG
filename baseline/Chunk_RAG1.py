import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from bert_score import score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import time
import re

# Replace 'YOUR_ACTUAL_API_KEY' with your actual OpenAI API key
client = openai.OpenAI(
    api_key="YOUR_ACTUAL_API_KEY"
)

input_file = "filtered_v1.0-simplified-nq-dev-all.jsonl"  # Ensure this is the correct filtered dataset
results = []
max_examples = 50  # Or a larger number for more comprehensive results
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_tokens(document_tokens):
    tokens = [t["token"] for t in document_tokens if not t["html_token"]]
    return " ".join(tokens)

def chunk_text(text, max_length=500):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def get_top_chunks(question, chunks, top_k=3):
    embeddings = model.encode([question] + chunks)
    question_vec = embeddings[0]
    chunk_vecs = embeddings[1:]
    similarities = cosine_similarity([question_vec], chunk_vecs)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def extract_ground_truth_short_answer(example):
    """
    Extracts the ground truth short answer from an NQ dataset example.

    Args:
        example (dict): A dictionary representing a single example from the NQ dataset.

    Returns:
        str: The extracted ground truth short answer, or None if not found.
    """
    try:
        annotations = example.get("annotations", [])
        if not annotations:
            print("Warning: Example has no 'annotations'.")
            return None

        first_annotation = annotations[0]  # Access the first annotation directly
        short_answers = first_annotation.get("short_answers", [])

        if not short_answers:
            print("Warning: First annotation has no 'short_answers'.")
            return None

        tokens = example["document_tokens"]
        if not tokens:
            print("Warning: Example has no 'document_tokens'.")
            return None

        extracted_answers = []
        for ans in short_answers:
            start = ans.get("start_token")
            end = ans.get("end_token")

            if start is None or end is None:
                print("Warning: short_answer missing 'start_token' or 'end_token'.")
                continue  # Skip this short answer

            try:
                text_parts = [t["token"] for t in tokens[start:end] if not t.get("html_token")]
                text = " ".join(text_parts).strip()
                extracted_answers.append(text)
            except IndexError:
                print(f"Warning: 'start_token' or 'end_token' out of bounds for 'document_tokens'. start: {start}, end: {end}, len(tokens): {len(tokens)}")
                continue # Skip to the next short answer

        if not extracted_answers:
            return None # Return None if no answers were extracted

        return " ".join(extracted_answers)  # Combine multiple short answers if present

    except KeyError as e:
        print(f"Error: Missing key in example: {e}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        return None

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def fuzzy_match(pred, truth, threshold=85):
    if not pred or not truth:
        return False
    pred_normalized = normalize_text(pred)
    truth_normalized = normalize_text(truth)
    score = fuzz.token_set_ratio(pred_normalized, truth_normalized)
    return score >= threshold

def generate_answer_with_gpt(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question below.

    Context:
    {context}

    Question:
    {question}

    Answer using only the information in the context. If the answer is not in the context, say "I don't know."
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2  # Consider lowering the temperature for more deterministic results
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error generating answer:", e)
        return "Error."

# --- Run Evaluation and Generate Table ---
with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if i >= max_examples:
            break
        try:
            example = json.loads(line)
            question = example["question_text"]
            doc_tokens = example["document_tokens"]
            text = extract_text_from_tokens(doc_tokens)
            chunks = chunk_text(text)
            top_chunks = get_top_chunks(question, chunks)

            generated_answer = generate_answer_with_gpt(question, top_chunks)
            ground_truth = extract_ground_truth_short_answer(example)  # Use the improved function

            if generated_answer and ground_truth:
                P, R, F1 = score([generated_answer], [ground_truth], lang="en")
                bert_precision = P.item()  # BERTScore precision (0 to 1)
                bert_recall = R.item()    # BERTScore recall (0 to 1)
                bert_score_f1 = F1.item()  # BERTScore F1 (0 to 1)

                fuzzy_result = fuzzy_match(generated_answer, ground_truth)
                precision = 1.0 if fuzzy_result else 0.0  # Fuzzy match precision (0 or 1)
                recall = 1.0 if fuzzy_result else 0.0     # Fuzzy match recall (0 or 1)
            else:
                bert_score_f1 = 0.0
                precision = 0.0
                recall = 0.0

            is_match = fuzzy_match(generated_answer, ground_truth)

            # Debugging and logging
            print(f"Question Number: {i + 1}")  # Print question number
            print(f"Generated: {generated_answer}")
            print(f"Ground Truth: {ground_truth}")
            print(f"BERTScore - P: {bert_precision}, R: {bert_recall}, F1: {bert_score_f1}")
            print(f"Fuzzy Match: {'YES' if is_match else 'NO'}")
            print(f"Fuzzy Precision: {precision}, Fuzzy Recall: {recall}")

            results.append({
                "Question": question,
                "Ground Truth": ground_truth,
                "ChunkRAG Answer": generated_answer,
                "BERTScore F1": round(bert_score_f1, 3),
                "Fuzzy Match": "YES" if is_match else "NO",
                "Precision": precision,  # Fuzzy match precision (0 or 1)
                "Recall": recall,      # Fuzzy match recall (0 or 1)
                "BERTScore Precision": round(bert_precision, 3),  # BERTScore precision (0 to 1)
                "BERTScore Recall": round(bert_recall, 3)         # BERTScore recall (0 to 1)
            })
            time.sleep(1)  # Add a delay to avoid rate limiting

        except openai.RateLimitError as e:
            print(f"Rate limit error in line {i + 1}: {e}")
            time.sleep(10)  # Wait and retry
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in line {i + 1}: {e}")
        except MemoryError as me:
            print(f"Memory error in line {i + 1}: {me}")
        except Exception as e:
            print(f"Error in line {i + 1}: {e}")

df = pd.DataFrame(results)
print("\nComparison Table:\n", df)
df.to_csv("chunkrag_comparison_results.csv", index=False)

# --- Visualization ---

# Extract question labels for x-axis
questions = [f"Q{i + 1}" for i in range(len(df))]  # Create question labels like Q1, Q2, ...

# Create subplots
fig = make_subplots(rows=3, cols=1, subplot_titles=("BERTScore Precision", "BERTScore Recall", "BERTScore F1"))

# Add Precision chart to subplot
fig.add_trace(go.Bar(x=questions, y=df['BERTScore Precision'], name='BERTScore Precision'), row=1, col=1)  # BERTScore precision (0 to 1)

# Add Recall chart to subplot
fig.add_trace(go.Bar(x=questions, y=df['BERTScore Recall'], name='BERTScore Recall'), row=2, col=1)     # BERTScore recall (0 to 1)

# Add F1-score chart to subplot
fig.add_trace(go.Bar(x=questions, y=df['BERTScore F1'], name='BERTScore F1'), row=3, col=1)           # BERTScore F1 (0 to 1)

# Update layout
fig.update_layout(height=800, width=1200, title_text="BERTScore Metrics (Questions 1-{})".format(len(questions)))

# Update x-axis labels
fig.update_xaxes(title_text="Questions", row=1, col=1)
fig.update_xaxes(title_text="Questions", row=2, col=1)
fig.update_xaxes(title_text="Questions", row=3, col=1)

# Update y-axis labels
fig.update_yaxes(title_text="Precision Score", row=1, col=1)
fig.update_yaxes(title_text="Recall Score", row=2, col=1)
fig.update_yaxes(title_text="F1 Score", row=3, col=1)

# Rotate x-axis labels
fig.update_xaxes(tickangle=-45)

fig.show()
