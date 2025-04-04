from kgrag import graph_utils
from utils.store_utils import Store
from visualizations.graph_visualizations import visualize_graph_topological, visualize_graph_with_cycle_detection
import re
from kgrag.graph_utils import NodeTagType

class Retriever:
    def __init__(self, embedding_dict, store_dict, verbose=False):
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
    def embed(self, corpus):
        '''
        1. Create knowledge graph from corpus
        2. For each root to leaf path create an embedding and store the data.
        Data Obj: {
            "path": str
            "data": str
        }
        '''

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
                path2text = ", ".join([re.sub(r'[\.|\?|!]', '', v.value) for v in path])
                path_data_list.append([path2text, u.data])
            else:
                for v in u.adj:
                    stack.append(path + [v])

        for path, data in path_data_list:
            path_tokens = self.tokenizer(path, language="english")
            if len(path_tokens) > 1:
                raise NotImplementedError(f"Path with multiple tokens.\n\t Path: {path}")
            path_embeddings = self.model.encode(path_tokens)
            self.store.write(path_embeddings, {"path": path, "data": data})

    def retrieve(self, query):
        #TODO: Do we need to tokenize the query?
        # q_tokens = self.tokenizer(query, language="english")
        q_embeddings = self.model.encode(query)
        retrieve_obj_list = self.store.nn_query(q_embeddings, 10)
        return [(obj["path"], obj["data"]) for obj in retrieve_obj_list]

    def close(self):
        self.store.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
