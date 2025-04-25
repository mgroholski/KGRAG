from utils.store_utils import Store
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import re

class Retriever:
    def __init__(self, embedding_dict, store_dict, agent, verbose=False):
        if "model" in embedding_dict and "tokenizer" in embedding_dict and "model_dim" in embedding_dict:
            self.model = embedding_dict["model"]
            self.tokenizer = embedding_dict["tokenizer"]
            self.model_dim = embedding_dict["model_dim"]
        else:
            raise Exception("Could not read embedding dictionary information. Please format correctly.")

        if "storepath" in store_dict:
            self.operation = store_dict["operation"]
            self.store = Store(self.model_dim, store_dict["storepath"], self.operation,verbose)
        else:
            raise Exception("Could not read store dictionary information. Please format correctly.")

        self.agent = agent
        self.verbose = verbose

        # Set up BM25 only if metadata exists
        self.all_chunks = self.store.metadata
        if self.all_chunks:
            self.bm25_corpus = [word_tokenize(chunk.lower()) for chunk in self.all_chunks]
            self.bm25_model = BM25Okapi(self.bm25_corpus)
        else:
            self.bm25_corpus = []
            self.bm25_model = None  # Delay initialization until retrieve()

    def embed(self, corpus):
        if self.operation == "r":
            return
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
        token_amount = 128
        filtered_chunks = []
        for chunk in combined:
            if hasattr(self.agent, "trim_context"):
                chunk = self.agent.trim_context([chunk])[0]

            prompt = f"""
            Query: {query}
            Chunk: {chunk}

            On a scale of 1 to 10, how relevant is this chunk to the query above?
            Only return a number from 1 to 10.

            IMPORTANT: Your answer MUST start with "<start_a>" and end with "</end_a>".
            For example, if asked "Who was the first president of the United States?", you must reply "<start_a>George Washington</end_a>".

            Make your response under or equal to {token_amount} tokens.
            """
            try:
                score = None
                retry_cnt = 0
                while score == None and retry_cnt < 10:
                    response = self.agent.ask(prompt, max_length=token_amount)
                    match = re.search(r'<start_a>(.*?)</end_a>', response)
                    if match:
                        score = int(match.group(1))
                    retry_cnt += 1

                if retry_cnt == 10 and score == None:
                    raise Exception(f"Could not get good LLM output format for {prompt}.")

                if score >= 6:
                    filtered_chunks.append(chunk)
            except Exception as e:
                print(f"LLM scoring failed for a chunk: {e}")
                continue

        return filtered_chunks

    def save(self):
        if self.operation != "r":
            self.store.write()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write()
