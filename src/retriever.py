from numpy import dot
from numpy.linalg import norm

class Retriever:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.store = []
        self.embeddings = []

    def embed(self, corpus, data):
        for text, data in zip(corpus, data):
            text_tokens = self.tokenizer(text, language="english")
            text_embedding = self.model.encode(text_tokens)
            if text_embedding.shape[0] > 1:
                raise NotImplementedError(f"Path embedding with multiple embedding vectors for \"{text}\"\n\nTokens: {text_tokens}\n\nShape:{text_embedding.shape}.")

            self.store.append((text, data))
            self.embeddings.append(text_embedding)

    def retrieve(self, query, std_multiplier=1.5):
        """
        Retrieve documents with scores above a significant threshold based on standard deviation.

        Args:
            query: The query string
            std_multiplier: How many standard deviations a drop needs to be to be considered significant
                           Higher values return more results, lower values return fewer

        Returns:
            List of (score, document) tuples above the threshold
        """
        query_tokens = self.tokenizer(query, language="english")
        query_embeddings = self.model.encode(query_tokens)

        # Calculate scores for all embeddings
        all_scores = []
        for idx, embedding in enumerate(self.embeddings):
            score = 0
            for q_embedding in query_embeddings:
                score += Retriever.compare_embeddings(embedding, q_embedding)
            score /= len(query_tokens)
            all_scores.append((score, self.store[idx]))

        # Sort scores in descending order
        all_scores.sort(reverse=True)

        if len(all_scores) <= 1:
            return all_scores

        # Calculate the differences between consecutive scores
        differences = [all_scores[i-1][0] - all_scores[i][0] for i in range(1, len(all_scores))]

        # Calculate mean and standard deviation of the differences
        mean_diff = sum(differences) / len(differences)
        std_diff = (sum((d - mean_diff) ** 2 for d in differences) / len(differences)) ** 0.5

        # Define the threshold for a significant drop
        significant_drop_threshold = mean_diff + std_multiplier * std_diff

        # Find where the significant drop occurs
        results = [all_scores[0]]
        for i in range(1, len(all_scores)):
            current_drop = all_scores[i-1][0] - all_scores[i][0]

            # If we encounter a significant drop, stop here
            if current_drop > significant_drop_threshold:
                break

            results.append(all_scores[i])

        return results

    @staticmethod
    def compare_embeddings(e1, e2):
        return (dot(e1, e2)) / (norm(e1) * norm(e2))
