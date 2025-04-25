from utils.store_utils import Store
import numpy as np
import re

class Retriever:
    def __init__(self, embedding_dict, store_dict, agent, verbose=False):
        if "model" in embedding_dict and "tokenizer" in embedding_dict and "model_dim" in embedding_dict:
            self.model = embedding_dict["model"]
            self.tokenizer = embedding_dict["tokenizer"]
            self.model_dim = embedding_dict["model_dim"]
        else:
            raise Exception("Could not read embedding dictionary information. Please format correctly.")

        # Initialize the storage
        if "storepath" in store_dict:
            self.operation = store_dict["operation"]
            self.store = Store(self.model_dim, store_dict["storepath"], self.operation, verbose)
        else:
            raise Exception("Could not read store dictionary information. Please format correctly.")

        self.agent = agent
        self.verbose = verbose
        self.all_tables = self.store.metadata

    def embed(self, corpus):
        if self.operation == "r":
            return
        table_embeddings = self.model.encode([corpus])
        self.store.write(table_embeddings, corpus)

    def retrieve(self, query) -> list[str]:
        query_embedding = self.model.encode([query])
        retrieved_tables = self.store.nn_query(query_embedding, 1)  # Top 5 nearest neighbors
        return retrieved_tables

    def close(self):
        if self.operation != "r":
            self.store.save()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
