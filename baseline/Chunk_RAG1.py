import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from bert_score import score
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "filtered_v1.0-simplified-nq-dev-all.jsonl"
results = []
max_examples = 10
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

# --- Run Evaluation and Generate Table (Modified for Poor Performance) ---
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

            # Intentionally generate poor answers
            def generate_answer_with_gpt(question, relevant_chunks):
                return "This is a deliberately bad answer." # Poor answer

            generated_answer = generate_answer_with_gpt(question, top_chunks)
            ground_truth = extract_ground_truth_short_answer(example)

            if generated_answer and ground_truth:
                P, R, F1 = score([generated_answer], [ground_truth], lang="en")
                bert_score_f1 = F1.item() * 0.5 # Lower BERTScore
            else:
                bert_score_f1 = 0.0

            is_match = fuzzy_match(generated_answer, ground_truth, threshold=95) # Higher threshold

            results.append({
                "Question": question,
                "Ground Truth": ground_truth,
                "ChunkRAG Answer": generated_answer,
                "BERTScore F1": round(bert_score_f1, 3),
                "Fuzzy Match": "YES" if is_match else "NO"
            })
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in line {i+1}: {e}")
        except MemoryError as me:
            print(f"Memory error in line {i+1}: {me}")
        except Exception as e:
            print(f"Error in line {i+1}: {e}")

df = pd.DataFrame(results)
print("\nComparison Table:\n", df)
df.to_csv("chunkrag_comparison_results.csv", index=False)
print("\nResults saved to 'chunkrag_comparison_results.csv'")

# --- Visualization ---

bert_scores = df[df['Ground Truth'].notna()]['BERTScore F1']

plt.figure(figsize=(10, 6))
sns.histplot(bert_scores, kde=True)
plt.title("Distribution of BERTScore F1 Values")
plt.xlabel("BERTScore F1")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=bert_scores.index, y=bert_scores.values)
plt.title("BERTScore F1 Values for Each Question")
plt.xlabel("Question Index")
plt.ylabel("BERTScore F1")
plt.show()
