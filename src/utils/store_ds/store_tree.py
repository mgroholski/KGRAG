import json
import faiss
import os
import numpy as np
import heapq
import threading
import torch

class StoreNode:
    def __init__(self, dim, text_value="", data=None, use_gpu=True):
        self.data = data
        self.text_value = text_value
        self.index = faiss.IndexFlatIP(dim)
        self.adj_dict = {} # Allows for the ability to get the next node based on the data.
        self.adj = []
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Move index to GPU if available and requested
        if self.use_gpu:
            self._move_index_to_gpu()

    def _move_index_to_gpu(self):
        """Move the FAISS index to GPU if available."""
        try:
            # Get number of available GPUs
            ngpus = faiss.get_num_gpus()
            if ngpus > 0:
                # Use the first GPU
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            else:
                self.use_gpu = False
        except Exception as e:
            print(f"Error moving index to GPU: {e}. Using CPU index.")
            self.use_gpu = False

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

    def __lt__(self, other):
        """Less than comparison, primarily used for heap operations"""
        if not isinstance(other, StoreNode):
            return NotImplemented
        return self.text_value < other.text_value

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, StoreNode):
            return NotImplemented
        return self.text_value == other.text_value and self.data == other.data

    @staticmethod
    def get_avg_embedding(q_embeddings):
        return np.mean(q_embeddings, axis=0, keepdims=True)

    @staticmethod
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

class StoreTree:
    def __init__(self, dim, path=None, read=False, verbose=False, use_gpu=True):
        '''
            Needs to store a list of page titles to begin the search.
            Root should not be included: "" (Root Node) -> Ringo Starr -> Songs
        '''

        self.folder_path = path
        self.verbose = verbose
        self.embedding_dim = dim
        self.write_lock = threading.Lock()
        self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            print(f"GPU is available. FAISS will use GPU acceleration.")
        elif use_gpu:
            print(f"GPU was requested but is not available. Using CPU.")

        if (read == "r" and self.folder_path is not None and os.path.exists(path)
            and os.path.exists(os.path.join(path, "tree_metadata.json"))):
            print(f"Loading StoreTree from {path}...")

            metadata_path = os.path.join(path, "tree_metadata.json")
            with open(metadata_path, "r") as read_file:
                tree_data = json.load(read_file)

            if tree_data["embedding_dim"] != dim:
                print(f"Warning: Stored embedding dimension ({tree_data['embedding_dim']}) differs from requested ({dim})")

            nodes_dict = {}
            for node_data in tree_data["nodes"]:
                node_id = node_data["id"]
                text_value = node_data["text_value"]
                data = node_data["data"]
                new_node = StoreNode(dim, text_value, data)
                node_index_path = os.path.join(path, f"node_{node_id}.index")
                if os.path.exists(node_index_path):
                    new_node.index = faiss.read_index(node_index_path)
                    if self.use_gpu:
                        new_node._move_index_to_gpu()
                nodes_dict[node_id] = new_node

            for node_data in tree_data["nodes"]:
                node_id = node_data["id"]
                node = nodes_dict[node_id]
                for child_info in node_data["children"]:
                    child_id = child_info["id"]
                    title = child_info["title"]
                    child_node = nodes_dict[child_id]
                    node.adj.append(child_node)
                    node.adj_dict[title] = child_node
            self.root = nodes_dict["root"]
            print("StoreTree loaded successfully")
        else:
            print("Creating new StoreTree...")
            self.root = StoreNode(dim, use_gpu=self.use_gpu)


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
                    new_node = StoreNode(self.embedding_dim, node_title, node_data, use_gpu=self.use_gpu)
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
            max_heap = [(0.0, "", self.root, [])]
            return_list = []

            gamma = 0.8
            scoring_function = lambda a : (-(sum([((gamma ** (len(new_score_list) - 1 - idx)) * i) for idx, i in enumerate(new_score_list)]) / sum([(gamma ** (len(new_score_list) - 1 - idx)) for idx in range(len(new_score_list))])))
            while max_heap and len(return_list) < k:
                avg_ip, nl_path, u, ip_list = heapq.heappop(max_heap)
                if len(u.text_value) > 0:
                    if len(nl_path):
                        nl_path += ", "
                    nl_path += f"{u.text_value}"

                if len(u.adj) > 0:
                    child_vals = u.search_children(q_embeddings)
                    for dist, node in child_vals:
                        new_score_list = ip_list + [dist]
                        try:
                            heapq.heappush(max_heap, (scoring_function(new_score_list), nl_path, node, new_score_list))
                        except Exception as e:
                            print("Heap: ", max_heap)
                            print("Bad Search: ", (scoring_function(new_score_list), nl_path, node, new_score_list))
                            raise e

                elif u.data != None:
                    return_list.append((-avg_ip, {"path": nl_path, "data": u.data}))

            # print("Retrieve List: ")
            # for (score,path) in return_list:
            #     print(f"\t Score: {score}\n\t Path:{path}")

            return [path_obj for (_, path_obj) in return_list]

    def range_query(self, q_embeddings, threshold=0.1):
        raise NotImplementedError()

    def close(self):
        if self.folder_path:
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)

            tree_data = {
                "embedding_dim": self.embedding_dim,
                "nodes": []
            }

            def traverse_tree(node, node_id, parent_id=None):
                node_data = {
                    "id": node_id,
                    "parent_id": parent_id,
                    "text_value": node.text_value,
                    "data": node.data,
                    "children": []
                }

                node_index_path = os.path.join(self.folder_path, f"node_{node_id}.index")
                # If using GPU, convert index back to CPU before saving
                index_to_save = node.index
                if node.use_gpu:
                    index_to_save = faiss.index_gpu_to_cpu(node.index)
                faiss.write_index(index_to_save, node_index_path)
                for i, (title, child_node) in enumerate(node.adj_dict.items()):
                    child_id = f"{node_id}_{i}"
                    node_data["children"].append({
                        "id": child_id,
                        "title": title
                    })
                    traverse_tree(child_node, child_id, node_id)

                tree_data["nodes"].append(node_data)

            traverse_tree(self.root, "root")
            metadata_path = os.path.join(self.folder_path, "tree_metadata.json")
            with open(metadata_path, "w") as write_file:
                json.dump(tree_data, write_file, indent=2)

            print(f"Tree data saved to {self.folder_path}")
        else:
            print("No write path provided...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
