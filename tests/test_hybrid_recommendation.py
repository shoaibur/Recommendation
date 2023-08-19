import unittest
import torch
from models.hybrid_recommendation import HybridRecommendation

class TestHybridRecommendation(unittest.TestCase):

    def setUp(self):
        # Sample data
        self.users_data = [
            [0.1, 0.2, 0.3],
            [0.5, 0.3, 0.1],
            [0.4, 0.6, 0.2]
        ]
        self.items_data = [
            [0.2, 0.3, 0.1],
            [0.6, 0.2, 0.4],
            [0.5, 0.4, 0.6],
            [0.1, 0.7, 0.3],
            [0.3, 0.1, 0.8]
        ]
        self.interaction_data = [
            [5, -1, 2, 3, 2],
            [1, 4, -1, -1, 3],
            [3, -1, -1, 1, -1]
        ]

        self.hybrid_rec = HybridRecommendation(self.users_data, self.items_data, self.interaction_data)
        self.hybrid_rec.train()
        self.scores = self.hybrid_rec.predict()
        self.top_2_items = self.hybrid_rec.get_top_k_items(k=2)

    def test_predict_output_shape(self):
        self.assertEqual(self.scores.shape, (len(self.users_data), len(self.items_data)))

    def test_alpha_range(self):
        self.assertTrue(0 <= self.hybrid_rec.alpha.item() <= 1)

    def test_top_k_items_shape(self):
        self.assertEqual(self.top_2_items.shape, (len(self.users_data), 2))

    def test_get_top_k_items(self):
        # Get the top 2 items for each user
        top_2_items = self.hybrid_rec.get_top_k_items(k=2)
        print(top_2_items)
        # Ensure it returns 2 items for each user
        self.assertEqual(top_2_items.shape, (3, 2))
        self.assertTrue(torch.equal(top_2_items, torch.tensor([[1, 0],[3, 2],[1, 4]])))

if __name__ == "__main__":
    unittest.main()
