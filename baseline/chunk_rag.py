import json
import re
import numpy as np
import openai
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# OpenAI client
client = OpenAI()

# Load filtered dataset
input_file = "filtered_v1.0-simplified-nq-dev-all.jsonl"
examples = []
with open(input_file, 'r') as f:
    for line in f:
        examples.append(json.loads(line))

# --- Utilities ---
def extract_text_from_tokens(document_tokens):
    tokens = [t["token"] for t in document_tokens if not t["html_token"]]
    return " ".join(tokens)

def chunk_text(text, max_length=500):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_top_chunks(question, chunks, top_k=3):
    embeddings = model.encode([question] + chunks)
    question_vec = embeddings[0]
    chunk_vecs = embeddings[1:]
    similarities = cosine_similarity([question_vec], chunk_vecs)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def score_chunk_with_gpt(question, chunk):
    prompt = f"""
You are an assistant helping score the relevance of information.

Question: "{question}"

Chunk: "{chunk}"

On a scale of 1 to 5, how relevant is this chunk for answering the question?

Only respond with a single number (1 to 5). Do not explain.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )
        score_text = response.choices[0].message.content.strip()
        return int(score_text)
    except Exception as e:
        print("Error scoring chunk:", e)
        return 0

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
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error generating answer:", e)
        return "Error."

def extract_ground_truth_short_answer(example):
    annotations = example.get("annotations", [])
    if not annotations:
        return None
    short_answers = annotations[0].get("short_answers", [])
    if not short_answers:
        return None
    tokens = example["document_tokens"]
    for ans in short_answers:
        start, end = ans["start_token"], ans["end_token"]
        text = " ".join(t["token"] for t in tokens[start:end] if not t["html_token"])
        return text.strip()
    return None

def fuzzy_match(pred, truth, threshold=85):
    if not pred or not truth:
        return False
    score = fuzz.token_set_ratio(pred, truth)
    return score >= threshold

# --- Run Evaluation Across Dataset ---
correct = 0
total = 0
max_examples = 10  # You can increase this safely once it works!

print("\nRunning ChunkRAG on", max_examples, "examples...\n")

for i, example in enumerate(examples[:max_examples]):
    try:
        question = example["question_text"]
        doc_tokens = example["document_tokens"]
        text = extract_text_from_tokens(doc_tokens)
        chunks = chunk_text(text)
        top_chunks = get_top_chunks(question, chunks)

        # Filter by GPT relevance score
        filtered_chunks = [
            chunk for chunk in top_chunks
            if score_chunk_with_gpt(question, chunk) >= 3
        ]

        if not filtered_chunks:
            continue  # Skip if no relevant chunks found

        answer = generate_answer_with_gpt(question, filtered_chunks)
        ground_truth = extract_ground_truth_short_answer(example)

        is_match = fuzzy_match(answer, ground_truth)

        print(f"\nExample {i+1}")
        print("Q:", question)
        print("GPT Answer:", answer)
        print("Ground Truth:", ground_truth)
        print("Fuzzy Match:", "YES" if is_match else "NO")

        total += 1
        correct += int(is_match)

    except Exception as e:
        print(f"Skipping example {i+1} due to error:", e)

# --- Final Results ---
accuracy = (correct / total) * 100 if total else 0
print(f"\nProcessed: {total} examples")
print(f"Correct Matches (Fuzzy): {correct}")
print(f"Accuracy: {accuracy:.2f}%")
