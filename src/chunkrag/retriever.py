from utils.store_utils import Store
import numpy as np
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
            chunk_embeddings = self.model.encode([chunk,])
            self.store.write(chunk_embeddings, chunk)

    def retrieve(self, query):
        #TODO: Implement hybrid retrieval and advanced filtering
        q_embedding = self.model.encode([query])
        retrieve_obj_list = self.store.nn_query(q_embedding, 10)
        return retrieve_obj_list

    def close(self):
        self.store.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
