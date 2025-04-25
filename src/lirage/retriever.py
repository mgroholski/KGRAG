import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from utils.store_utils import Store

class LIRAGERetriever:
    def __init__(self, embedding_info, store_info, agent=None, verbose=False):
        self.verbose = verbose
        self.agent = agent
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.retriever = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to(self.device).eval()
        self.ret_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.reader = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large").to(self.device).eval()
        self.read_tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large")


        self.store = Store(embedding_info['model_dim'], store_info["storepath"], verbose)
        self.tables = []
        self.table_embs = []

    def embed(self, corpus):

        flat_text = corpus.strip()
        self.tables.append(flat_text)

        table_embedding = self.retriever(
            **self.ret_tokenizer(
                flat_text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            ).to(self.device)
        ).last_hidden_state.squeeze(0)

        self.table_embs.append(table_embedding)
        self.store.write(table_embedding.detach().cpu().numpy(), flat_text)

    def retrieve(self, q, k=5):
        q_emb = self.retriever(
            **self.ret_tokenizer(
                q,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=32
            ).to(self.device)
        ).last_hidden_state.squeeze(0)
        # normalize query embeddings
        q_emb = q_emb / torch.norm(q_emb, dim=1, keepdim=True)

        scores = []
        for table in self.table_embs:
            table = table / torch.norm(table, dim=1, keepdim=True)
            sim = torch.matmul(q_emb, table.T)
            score = torch.max(sim, dim=1).values.sum().item()
            scores.append(score)

        top_k_idx = np.argpartition(scores, -k)[-k:]
        return [self.tables[i] for i in top_k_idx]


    def close(self):
        if self.operation != "r":
            self.store.save()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
