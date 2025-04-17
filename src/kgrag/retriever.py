from kgrag import graph_utils
from utils.store_utils import Store
from visualizations.graph_visualizations import visualize_graph_topological, visualize_graph_with_cycle_detection
import re
import numpy as np
from kgrag.graph_utils import NodeTagType

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

        self.visualize = verbose
        self.agent = agent

    def embed(self, corpus):
        page_graph = graph_utils.extract_relationship_graphs(corpus)
        if self.visualize:
            try:
                visualize_graph_topological(page_graph)
            except Exception as e:
                print(e)
                visualize_graph_with_cycle_detection(page_graph)

        path_data_list = []
        stack = [[page_graph]]
        while stack:
            path = stack.pop()
            u = path[-1]

            if not len(u.adj) and u.type == NodeTagType.TH:
                path2text = ", ".join([v.value for v in path])
                path_data_list.append([path2text, u.data])
            else:
                for v in u.adj:
                    stack.append(path + [v])

        for path, data in path_data_list:
            path_embeddings = self.model.encode([path])
            self.store.write(path_embeddings, {"path": path, "data": data})

    def retrieve(self, query):
        
        std_lambda = 1.0

        q_embeddings = self.model.encode([query])
        retrieve_obj_list = self.store.nn_query(q_embeddings, 100)

        
        paths = [obj["path"] for obj in retrieve_obj_list]

        # Recompute embeddings for each path
        path_embeddings = self.model.encode(paths)

        
        sims = [self.cos_sim(q_embeddings[0], pe) for pe in path_embeddings]
        top_score = max(sims)
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)

        
        filtered = [(obj, sim) for obj, sim in zip(retrieve_obj_list, sims) if sim >= top_score - std_lambda * std_sim]
        filtered_paths = [f[0]["path"] for f in filtered]

        
        lcs = self._longest_common_subsequence(filtered_paths)
        print(f"LCS: {lcs}")

        if len(lcs.split()) >= 3:
            cleaned_paths = [(obj["path"].replace(lcs, '').strip(), obj["data"]) for obj, _ in filtered]
        else:
            cleaned_paths = [(obj["path"], obj["data"]) for obj, _ in filtered]

        # Rerank cleaned paths
        cleaned_texts = [p for p, _ in cleaned_paths]
        cleaned_embeddings = self.model.encode(cleaned_texts)
        scores = [self.cos_sim(q_embeddings[0], emb) for emb in cleaned_embeddings]

        ranked = sorted(zip(cleaned_paths, scores), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:10]]

    def _longest_common_subsequence(self, strings):
        def lcs(a, b):
            dp = [["" for _ in range(len(b)+1)] for _ in range(len(a)+1)]
            for i in range(len(a)):
                for j in range(len(b)):
                    if a[i] == b[j]:
                        dp[i+1][j+1] = dp[i][j] + a[i]
                    else:
                        dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1], key=len)
            return dp[-1][-1]

        if not strings:
            return ""
        lcs_result = strings[0]
        for s in strings[1:]:
            lcs_result = lcs(lcs_result, s)
            if not lcs_result:
                break
        return lcs_result

    def cos_sim(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

    def close(self):
        self.store.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
