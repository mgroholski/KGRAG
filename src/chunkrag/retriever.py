from utils.store_utils import Store
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

class Retriever:
    def __init__(self, embedding_dict, store_dict, agent, verbose=False):
        if "model" in embedding_dict and "tokenizer" in embedding_dict and "model_dim" in embedding_dict:
            self.model = embedding_dict["model"]
            self.tokenizer = embedding_dict["tokenizer"]
            self.model_dim = embedding_dict["model_dim"]
        else:
            raise Exception("Could not read embedding dictionary information. Please format correctly.")

        if "storepath" in store_dict:
            self.store = Store(self.model_dim, store_dict["storepath"], verbose)
        else:
            raise Exception("Could not read store dictionary information. Please format correctly.")

        self.agent = agent

        # Set up BM25 only if metadata exists
        self.all_chunks = self.store.metadata
        if self.all_chunks:
            self.bm25_corpus = [word_tokenize(chunk.lower()) for chunk in self.all_chunks]
            self.bm25_model = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_corpus = []
            self.bm25_model = None  # Delay initialization until retrieve()

    def embed(self, corpus):
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        corpus_tokens = self.tokenizer(corpus, language="english")

        chunks = []
        cur_chunk = None
        for token in corpus_tokens:
            if not cur_chunk:
                cur_chunk = token
            else:
                cur_chunk_embedding = self.model.encode(cur_chunk)
                token_embedding = self.model.encode(token)
                if cosine_similarity(cur_chunk_embedding, token_embedding) >= 0.7 and len(cur_chunk) + len(token) + 1 < 500:
                    cur_chunk += f" {token}"
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = token
        chunks.append(cur_chunk)

        for chunk in chunks:
            chunk_embeddings = self.model.encode([chunk])
            self.store.write(chunk_embeddings, chunk)

    def retrieve(self, query) -> list[str]:
        # Lazy BM25 init if not yet done
        if self.bm25_model is None and self.store.metadata:
            self.all_chunks = self.store.metadata
            self.bm25_corpus = [word_tokenize(chunk.lower()) for chunk in self.all_chunks]
            self.bm25_model = BM25Okapi(self.bm25_corpus)

        # Dense retrieval (top 5)
        q_embedding = self.model.encode([query])
        dense_results = self.store.nn_query(q_embedding, 5)

        # Sparse retrieval (BM25, top 5)
        sparse_results = []
        if self.bm25_model:
            sparse_scores = self.bm25_model.get_scores(word_tokenize(query.lower()))
            sparse_ranked = sorted(zip(sparse_scores, self.all_chunks), reverse=True)[:5]
            sparse_results = [chunk for _, chunk in sparse_ranked]

        # Combine and deduplicate
        combined = []
        seen = set()
        for chunk in dense_results + sparse_results:
            if chunk not in seen:
                combined.append(chunk)
                seen.add(chunk)

        # LLM-based relevance filtering
        filtered_chunks = []
        for chunk in combined:
            prompt = f"""
Query: {query}
Chunk: {chunk}

On a scale of 1 to 10, how relevant is this chunk to the query above?
Only return a number from 1 to 10.
"""
            try:
                response = self.agent.ask(prompt, max_length=10)
                score = int(''.join(filter(str.isdigit, response)))
                if score >= 6:
                    filtered_chunks.append(chunk)
            except Exception as e:
                print(f"LLM scoring failed for a chunk: {e}")
                continue

        return filtered_chunks

    def close(self):
        self.store.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
