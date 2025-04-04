import json
import faiss
import os
import numpy as np

class Store:
    def __init__(self, dim, path=None, verbose=False):
        self.folder_path = path
        self.verbose = verbose
        if (path != None and os.path.exists(path)
        and os.path.exists(os.path.join([path, "embeddings.index"]))
        and os.path.exists(os.path.join([path, "metadata.json"]))):
            print("Loading from the provided path...")
            self.index = faiss.read_index(os.path.join([path, "embeddings.index"]))
            with open(os.path.join([path, "metadata.json"]), "r") as read_file:
                self.metadata = json.loads(read_file.read())
        else:
            print("Either did not provide a path or could not find the correct files... Store new store files.")
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def write(self, embedding, data):
        self.index.add(embedding)
        self.metadata.append(data)

    def nn_query(self, q_embedding, k=10):
        '''
        Parameters:
            q_embedding: The query embedding.
            k: The amount of nearest neighbors to return.
        Return:
            A list of chunks.
        '''

        distances, indices = self.index.search(q_embedding, k)
        return self._extract_chunks(distances, indices)

    def range_query(self, q_embedding, threshold=0.1):
        '''
        Parameters:
            q_embedding: The query embedding.
            threshold: The allowable threshold to return.
        Return:
            A list of chunks.
        '''
        distances, indices = self.index.range_search(q_embedding, threshold)
        return self._extract_chunks(distances, indices)

    def _extract_chunks(self, distances, indices):
        retrieve_chunks = []
        for dist, idx in zip(distances, indices):
            data = self.metadata[idx]
            if self.verbose:
                print("Distance: ", dist, "\n", "Data: ", data, "\n")
            retrieve_chunks.append(data)

        return retrieve_chunks


    def __exit__(self):
        if self.path:
            faiss_path = os.path.join([self.folder_path, "embeddings.index"])
            faiss.write_index(self.index, faiss_path)
            metadata_path = os.path.join([self.folder_path, "metadata.json"])
            with open(metadata_path, "w") as write_file:
                write_file.write(json.dumps(self.metadata))
        else:
            print("No write path provided...")

    @staticmethod
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm
