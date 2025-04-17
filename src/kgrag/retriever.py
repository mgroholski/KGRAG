from kgrag import graph_utils
# from utils.store_utils import Store
from utils.store_utils import StoreTree
from visualizations.graph_visualizations import visualize_graph_topological, visualize_graph_with_cycle_detection
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
            self.store = StoreTree(self.model_dim, None, False)
            # self.store = Store(self.model_dim, store_dict["storepath"], verbose)
        else:
            raise Exception("Could not read store dictionary information. Please format correctly.")

        self.visualize = verbose
        self.agent = agent

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

            if not len(u.adj) and (u.type == NodeTagType.TH or u.type == NodeTagType.ARRAY_TABLE):
                # path2text = ", ".join([v.value for v in path])
                path_data_list.append([path, u.data])
            else:
                for v in u.adj:
                    stack.append(path + [v])

        for path, _ in path_data_list:
            metadata = [{"title": u.value, "data": None if not len(u.data) else u.data} for u in path]
            title_embeddings = self.model.encode([a["title"] for a in metadata])
            self.store.write(title_embeddings, metadata)

    def retrieve(self, query):
        '''
            Create a layered vector store like a Trie. Compare each layer within the to the query.
        '''
        # q_tokens = self.tokenizer(query, language="english")
        q_embeddings = self.model.encode([query])
        retrieve_obj_list = self.store.nn_query(q_embeddings, 3)
        return [(obj["path"], obj["data"]) for obj in retrieve_obj_list]

    def close(self):
        self.store.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
