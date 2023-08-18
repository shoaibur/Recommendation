import unittest
import torch

from models.deep_learning_based_recommendation import DeepLearningRec

class TestDeepLearningRec(unittest.TestCase):
    def setUp(self):
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
        self.dlr = DeepLearningRec(self.users_data, self.items_data, self.interaction_data)

    def test_trainability(self):
        try:
            self.dlr.train()
        except Exception as e:
            self.fail(f"Training failed with {e}")

    def test_prediction_shape(self):
        scores = self.dlr.predict()
        self.assertEqual(scores.shape, (len(self.users_data), len(self.items_data)))

    def test_get_top_k_items(self):
        # Get the top 2 items for each user
        top_2_items = self.dlr.get_top_k_items(k=2)
        print(top_2_items)
        # Ensure it returns 2 items for each user
        self.assertEqual(top_2_items.shape, (3, 2))
        self.assertTrue(torch.equal(top_2_items, torch.tensor([[1, 0],[2, 3],[1, 2]])))


if __name__ == "__main__":
    unittest.main()
