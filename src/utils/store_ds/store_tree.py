import json
import faiss
import os
import numpy as np
import heapq
import threading

class StoreNode:
    def __init__(self, dim, text_value="", data = None):
        self.data = data
        self.text_value = text_value
        self.index = faiss.IndexFlatIP(dim)
        self.adj_dict = {} # Allows for the ability to get the next node based on the data.
        self.adj = []

    def add_child(self, embedding, data, node):
        self.index.add(embedding)
        self.adj.append(node)
        self.adj_dict[data] = node

    def search_children(self, embeddings):
        avg_embedding = StoreNode.get_avg_embedding(embeddings)
        D, I = self.index.search(StoreNode.normalize_vector(avg_embedding), len(self.adj))
        idx = 0
        distances, indices = (D[idx], I[idx])
        res = []
        for dist, idx in zip(distances, indices):
            node = self.adj[idx]
            res.append((dist, node))
        return res

    @staticmethod
    def get_avg_embedding(q_embeddings):
        return np.mean(q_embeddings, axis=0, keepdims=True)

    @staticmethod
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

class StoreTree:
    def __init__(self, dim, path=None, verbose=False):
        '''
            Needs to store a list of page titles to begin the search.
            Root should not be included: "" (Root Node) -> Ringo Starr -> Songs
        '''

        self.folder_path = path
        self.verbose = verbose
        if (path != None):
            raise NotImplementedError("Implement reading in embeddings files.")

        self.embedding_dim = dim
        self.root = StoreNode(dim)
        self.write_lock = threading.Lock()


    def write(self, path_embeddings, path_metadata):
        '''
            Parameters:
                path_embeddings: A np array containing the embeddings of the path_data.
                path_data: A list of data that is associated with the embedding with the same index.
        '''
        with self.write_lock:
            u = self.root
            while path_metadata:
                data = path_metadata.pop(0)
                node_title = data["title"]
                node_data = data["data"]

                # Pops the first embedding
                embedding = path_embeddings[0]
                path_embeddings = path_embeddings[1:]
                if node_title in u.adj_dict:
                    u = u.adj_dict[node_title]
                else:
                    new_node = StoreNode(self.embedding_dim, node_title, node_data)
                    u.add_child(np.array([embedding]), node_title, new_node)
                    u = new_node

    def nn_query(self, q_embeddings, k=10):
        '''
            1. Initialize priority queue with root.
            2. For each child node, calculate average inner product.
            3. Add node to queue ranked by avg inner product.

            Concerns
                - We're searching through the entire tree. It may be worthwhile to optimize this by searching with range_query.
                To use range_query, we'd need to answer "Is there some threshold such that we don't care about answers below?"
        '''
        # Track current avg IP, node, natural language list, list of IPs
        max_heap = [(0.0, self.root, "", [])]
        leaf_list = []
        while max_heap:
            avg_ip, u, nl_path, ip_list = heapq.heappop(max_heap)
            if len(u.text_value) > 0:
                if len(nl_path):
                    nl_path += ", "
                nl_path += f"{u.text_value}"

            if len(u.adj) > 0:
                child_vals = u.search_children(q_embeddings)
                for dist, node in child_vals:
                    new_score_list = ip_list + [dist]
                    heapq.heappush(max_heap, (-(sum(new_score_list) / len(new_score_list)), node, nl_path, new_score_list))
            elif u.data != None:
                heapq.heappush(leaf_list, (avg_ip, {"path": nl_path, "data": u.data}))

        return_list = []
        while len(return_list) < k and len(leaf_list):
            return_list.append(heapq.heappop(leaf_list))

        return [path_obj for (_, path_obj) in return_list]

    def range_query(self, q_embeddings, threshold=0.1):
        raise NotImplementedError()

    def close(self):
        if self.path != None:
            raise NotImplementedError("Implement writing StoreTrees files.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
