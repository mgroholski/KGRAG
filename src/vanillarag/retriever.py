from utils.store_utils import Store

class VanillaRetriever:
    def __init__(self, embedding_dict, store_dict, agent, verbose=False):
        if "model" in embedding_dict and "tokenizer" in embedding_dict and "model_dim" in embedding_dict:
            self.model = embedding_dict["model"]
            self.tokenizer = embedding_dict["tokenizer"]
            self.model_dim = embedding_dict["model_dim"]
        else:
            raise Exception("Could not read embedding dictionary information. Please format correctly.")

        if "storepath" in store_dict:
            self.operation = store_dict["operation"]
            self.store = Store(self.model_dim, store_dict["storepath"], self.operation, verbose)
        else:
            raise Exception("Could not read store dictionary information. Please format correctly.")

        self.agent = agent
        self.verbose = verbose

    def embed(self, corpus):
        corpus_sentences = self.tokenizer(corpus, language="english")

        chunks = []
        cur_chunk = []
        word_limit = 100

        for sentence in corpus_sentences:
            words = sentence.split()
            if len(cur_chunk) + len(words) <= word_limit:
                cur_chunk.extend(words)
            else:
                chunks.append(" ".join(cur_chunk))
                cur_chunk = words

        if cur_chunk:
            chunks.append(" ".join(cur_chunk))

        for chunk in chunks:
            chunk_embedding = self.model.encode([chunk,])
            self.store.write(chunk_embedding, chunk)

    def retrieve(self, query) -> list:
        '''
        Retrieve top-10 most similar passages for the given query.
        '''
        q_embedding = self.model.encode([query])
        retrieve_obj_list = self.store.nn_query(q_embedding, 3)
        return retrieve_obj_list

    def close(self):
        if self.operation != "r":
            self.store.save()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
