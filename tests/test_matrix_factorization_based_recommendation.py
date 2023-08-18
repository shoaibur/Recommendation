import unittest
import torch

from models.matrix_factorization_based_recommendation import MatrixFactorization # Replace `your_module_name` with the name of the module containing the MatrixFactorization class

class TestMatrixFactorization(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.interaction_data = [
            [5, -1, 2, 3, 2],
            [1, 4, -1, -1, 3],
            [3, -1, -1, 1, -1]
        ]
        self.mf = MatrixFactorization(self.interaction_data)

    def test_initialization(self):
        # Test the dimensions of user and item factors
        self.assertEqual(self.mf.user_factors.shape, (3, 3))
        self.assertEqual(self.mf.item_factors.shape, (5, 3))

        # Test known interactions mask
        self.assertTrue(torch.equal(self.mf.known_interactions_mask, torch.tensor([[1, 0, 1, 1, 1], [1, 1, 0, 0, 1], [1, 0, 0, 1, 0]], dtype=torch.uint8)))

    def test_train_and_predict(self):
        # Train the model
        self.mf.train()

        # Get the predicted scores
        scores = self.mf.predict()

        # Ensure the predicted scores have the same dimensions as the interaction_data
        self.assertEqual(scores.shape, (3, 5))

    def test_get_top_k_items(self):
        # Get the top 2 items for each user
        top_2_items = self.mf.get_top_k_items(k=2)

        # Ensure it returns 2 items for each user
        self.assertEqual(top_2_items.shape, (3, 2))
        self.assertTrue(torch.equal(top_2_items, torch.tensor([[1, 0],[3, 2],[1, 4]])))

if __name__ == "__main__":
    unittest.main()
