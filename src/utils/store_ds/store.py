import json
import faiss
import os
import numpy as np

class Store:
    def __init__(self, dim, path=None, read=False, verbose=False):
        self.folder_path = path
        self.verbose = verbose
        if (read == "r" and path != None and os.path.exists(path)
        and os.path.exists(os.path.join(path, "embeddings.index"))
        and os.path.exists(os.path.join(path, "metadata.json"))):
            print("Loading from the provided path...")
            self.index = faiss.read_index(os.path.join(path, "embeddings.index"))
            with open(os.path.join(path, "metadata.json"), "r") as read_file:
                self.metadata = json.loads(read_file.read())
        else:
            print("Either did not provide a path or could not find the correct files... Creating new store files.")
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []

    def write(self, embedding, data):
        self.index.add(embedding)
        self.metadata.append(data)

    def nn_query(self, q_embeddings, k=10):
        '''
        Parameters:
            q_embeddings: The query embedding.
            k: The amount of nearest neighbors to return.
        Return:
            A list of chunks.
        '''
        avg_embedding = self._get_avg_embedding(q_embeddings)
        D, I = self.index.search(Store.normalize_vector(avg_embedding), k)
        idx = 0
        distances, indices = (D[idx], I[idx])
        return self._extract_chunks(distances, indices)

    def range_query(self, q_embeddings, threshold=0.1):
        '''
        Parameters:
            q_embeddings: The query embeddings.
            threshold: The allowable threshold to return.
        Return:
            A list of chunks.
        '''
        # Information on function: https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes
        avg_embedding = self._get_avg_embedding(q_embeddings)
        lims, D, I = self.index.range_search(Store.normalize_vector(avg_embedding), threshold)
        idx = 0
        distances, indices = (D[lims[idx]:lims[idx+1]], I[lims[idx]:lims[idx+1]])
        return self._extract_chunks(distances, indices)

    def _get_avg_embedding(self, q_embeddings):
        return np.mean(q_embeddings, axis=0, keepdims=True)

    def _extract_chunks(self, distances, indices):
        retrieve_chunks = []
        for dist, idx in zip(distances, indices):
            data = self.metadata[idx]
            if self.verbose:
                print("Distance: ", dist, "\n", "Data: ", data, "\n")
            retrieve_chunks.append(data)

        return retrieve_chunks

    def close(self):
        if self.folder_path:
            if not os.path.exists(self.folder_path):
                os.makedirs(self.folder_path)
            faiss_path = os.path.join(self.folder_path, "embeddings.index")
            faiss.write_index(self.index, faiss_path)
            metadata_path = os.path.join(self.folder_path, "metadata.json")
            with open(metadata_path, "w") as write_file:
                write_file.write(json.dumps(self.metadata))
        else:
            print("No write path provided...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm
