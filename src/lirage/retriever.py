import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from utils.store_ds.store_mat import StoreMat
from bs4 import BeautifulSoup
from kgrag.graph_utils import extract_first_level_tags

class LIRAGERetriever:
    def __init__(self, embedding_info, store_info, agent=None, verbose=False):
        self.verbose = verbose
        self.agent = agent
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.retriever = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to(self.device).eval()
        self.ret_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embedding_dim = self.retriever.config.hidden_size

        self.reader = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large").to(self.device).eval()
        self.read_tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large")

        self.store = StoreMat(self.embedding_dim, store_info["storepath"], store_info["operation"], verbose)
        self.operation = store_info["operation"]


    def embed(self, corpus):
        # Corpus is a HTML document
        corpus_soup = BeautifulSoup(corpus, "lxml").html.body

        # filters the html for all first-level tables
        tables = extract_first_level_tags(corpus_soup, ["table"])
        # filters the html for all first-level tables
        for table in tables:
            flat_text = str(table)

            table_embedding = self.retriever(
                **self.ret_tokenizer(
                    flat_text,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=512
                ).to(self.device)
            ).last_hidden_state.squeeze(0)

            t_emb = table_embedding.detach().cpu().numpy()
            self.store.write(t_emb, flat_text)

    def retrieve(self, q, k=3):
        q_emb = self.retriever(
            **self.ret_tokenizer(
                q,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=32
            ).to(self.device)
        ).last_hidden_state.squeeze(0)

        q_emb_np = q_emb.detach().cpu().numpy()
        return self.store.nn_query(q_emb_np, k=k)

    def close(self):
        if self.operation != "r":
            self.store.save()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
