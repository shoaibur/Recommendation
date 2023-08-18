import unittest
import torch

from models.content_based_recommendation_bert import ContentBasedRecommendation  # Update this import based on your directory structure

class TestContentBasedRecommendation(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.users_data = ["I love science fiction movies.", "I prefer romantic movies."]
        self.items_data = ["A science fiction movie about space.", "A romantic movie set in Paris.", "An action-packed thriller."]

        # Instantiate the recommender
        self.recommender = ContentBasedRecommendation(self.users_data, self.items_data)

    def test_get_embedding(self):
        # We're testing if embeddings are generated and if their shape matches what's expected
        embeddings = self.recommender._get_embedding(self.users_data)
        self.assertEqual(embeddings.size(0), len(self.users_data))
        self.assertEqual(embeddings.size(1), self.recommender.bert.config.hidden_size)

    def test_recommend(self):
        # We're testing if the similarity scores are generated and if their shape matches what's expected
        scores = self.recommender.recommend()
        self.assertEqual(scores.size(0), len(self.users_data))
        self.assertEqual(scores.size(1), len(self.items_data))

if __name__ == "__main__":
    unittest.main()
