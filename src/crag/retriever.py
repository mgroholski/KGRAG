import torch
import requests
import numpy as np
from enum import Enum
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils.store_ds.store import Store

class ActionType(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"

class RetrievalEvaluator:
    """
    Lightweight retrieval evaluator to assess the quality of retrieved documents.
    Based on T5-large model as described in the CRAG paper.
    """
    def __init__(self, model_path="t5-large", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        self.upper_threshold = 0.59  # Threshold for CORRECT action
        self.lower_threshold = -0.99  # Threshold for INCORRECT action

    def evaluate(self, query, documents):
        """
        Evaluate the relevance of documents to the query.

        Args:
            query: The input query.
            documents: List of retrieved documents.

        Returns:
            List of relevance scores for each document.
        """
        scores = []

        for document in documents:
            input_text = f"Question: {query} Document: {document}"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

            # Generate relevance score (from -1 to 1)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=2,  # Just need a yes/no response
                    return_dict_in_generate=True,
                    output_scores=True
                )

                # Simplified scoring mechanism - in practice would use proper fine-tuning
                # This is a placeholder for the actual model's behavior
                logits = outputs.scores[0]
                yes_score = logits[0, self.tokenizer.encode("yes")[0]].item()
                no_score = logits[0, self.tokenizer.encode("no")[0]].item()
                score = (yes_score - no_score) / 2  # Scale to -1 to 1
                scores.append(score)

        return scores

    def determine_action(self, scores):
        """
        Determine the action based on the relevance scores.

        Args:
            scores: List of relevance scores for each document.

        Returns:
            ActionType: The determined action.
        """
        if any(score > self.upper_threshold for score in scores):
            return ActionType.CORRECT
        elif all(score < self.lower_threshold for score in scores):
            return ActionType.INCORRECT
        else:
            return ActionType.AMBIGUOUS


class WebSearcher:
    """
    Component for searching the web to find relevant information.
    """
    def __init__(self, api_key, search_engine="google"):
        self.api_key = api_key
        self.search_engine = search_engine

    def rewrite_query(self, query):
        """
        Rewrite the query into keywords for web search.

        Args:
            query: The original query.

        Returns:
            str: Rewritten query with keywords.
        """
        # This would typically use an LLM to extract keywords
        # Simplified implementation for demonstration
        keywords = query.lower().split()
        keywords = [k for k in keywords if k not in ["what", "who", "when", "where", "is", "the", "a", "an"]]
        return ", ".join(keywords[:3])  # Take top 3 keywords

    def search(self, query):
        """
        Search the web for the query.

        Args:
            query: The search query.

        Returns:
            List of search results (documents).
        """
        rewritten_query = self.rewrite_query(query)

        # This would typically call a search API
        # Simplified implementation for demonstration
        if self.search_engine == "google":
            # In a real implementation, this would be an API call
            search_results = [
                f"Search result for {rewritten_query} - Document 1",
                f"Search result for {rewritten_query} - Document 2",
                f"Search result for {rewritten_query} - Document 3"
            ]

        return search_results


class KnowledgeProcessor:
    """
    Component for refining knowledge from retrieved documents.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def decompose(self, documents):
        """
        Decompose documents into knowledge strips.

        Args:
            documents: List of documents to decompose.

        Returns:
            List of knowledge strips.
        """
        strips = []

        for document in documents:
            # Split document into sentences
            sentences = document.split('.')

            # Create strips with 1-2 sentences
            current_strip = []
            for sentence in sentences:
                if sentence.strip():  # Skip empty sentences
                    current_strip.append(sentence.strip())

                    if len(current_strip) == 2:  # Create a strip every 2 sentences
                        strips.append('. '.join(current_strip) + '.')
                        current_strip = []

            # Add any remaining sentences
            if current_strip:
                strips.append('. '.join(current_strip) + '.')

        return strips

    def filter_strips(self, query, strips):
        """
        Filter knowledge strips based on relevance to the query.

        Args:
            query: The input query.
            strips: List of knowledge strips to filter.

        Returns:
            List of relevant knowledge strips.
        """
        scores = self.evaluator.evaluate(query, strips)

        # Filter strips with positive scores
        relevant_strips = [strip for strip, score in zip(strips, scores) if score > -0.5]

        return relevant_strips

    def recompose(self, strips):
        """
        Recompose filtered knowledge strips.

        Args:
            strips: List of relevant knowledge strips.

        Returns:
            Recomposed knowledge document.
        """
        return ' '.join(strips)

    def refine(self, query, documents):
        """
        Refine documents into relevant knowledge.

        Args:
            query: The input query.
            documents: List of documents to refine.

        Returns:
            Refined knowledge.
        """
        strips = self.decompose(documents)
        relevant_strips = self.filter_strips(query, strips)
        refined_knowledge = self.recompose(relevant_strips)

        return refined_knowledge


class CRAGRetriever:
    """
    Corrective Retrieval Augmented Generation (CRAG) retriever.
    """
    def __init__(self, embedding_dict, store_dict, agent, api_key, verbose=False):
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

        # CRAG components
        self.evaluator = RetrievalEvaluator()
        self.knowledge_processor = KnowledgeProcessor(self.evaluator)
        self.web_searcher = WebSearcher(api_key)

    def embed(self, corpus):
        """
        Embed corpus and store embeddings.

        Args:
            corpus: Text corpus to embed.
        """
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

    def vanilla_retrieve(self, query):
        """
        Standard retrieval method without corrections.

        Args:
            query: The input query.

        Returns:
            List of retrieved documents.
        """
        q_embedding = self.model.encode([query])
        retrieve_obj_list = self.store.nn_query(q_embedding, 3)
        return [obj.text for obj in retrieve_obj_list]

    def retrieve(self, query):
        """
        CRAG retrieval with corrections.

        Args:
            query: The input query.

        Returns:
            dict: Retrieval results with action type and knowledge.
        """
        # Step 1: Retrieve documents using standard retrieval
        documents = self.vanilla_retrieve(query)

        # Step 2: Evaluate documents and determine action
        scores = self.evaluator.evaluate(query, documents)
        action = self.evaluator.determine_action(scores)

        result = {
            "action": action,
            "knowledge": None
        }

        # Step 3: Process based on action
        if action == ActionType.CORRECT:
            # Refine knowledge from retrieved documents
            result["knowledge"] = self.knowledge_processor.refine(query, documents)

        elif action == ActionType.INCORRECT:
            # Search web for external knowledge
            web_results = self.web_searcher.search(query)
            result["knowledge"] = self.knowledge_processor.refine(query, web_results)

        else:  # AMBIGUOUS
            # Combine both internal and external knowledge
            internal_knowledge = self.knowledge_processor.refine(query, documents)
            web_results = self.web_searcher.search(query)
            external_knowledge = self.knowledge_processor.refine(query, web_results)

            result["knowledge"] = f"{internal_knowledge} {external_knowledge}"

        return result
