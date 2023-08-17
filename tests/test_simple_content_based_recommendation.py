import unittest
import numpy as np

class TestSimpleContentBasedRec(unittest.TestCase):

    def setUp(self):
        self.users_data = np.array([
            [0.1, 0.2, 0.3],
            [0.5, 0.3, 0.1],
            [0.4, 0.6, 0.2]
        ])
        self.items_data = np.array([
            [0.2, 0.3, 0.1],
            [0.6, 0.2, 0.4],
            [0.5, 0.4, 0.6],
            [0.1, 0.7, 0.3],
            [0.3, 0.1, 0.8]
        ])
        self.recommender = SimpleContentBasedRec(self.users_data, self.items_data)

    def test_scaling(self):
        # The mean of scaled data should be close to 0 and std deviation should be close to 1
        self.assertTrue(np.isclose(self.recommender.users_data.mean(), 0))
        self.assertTrue(np.isclose(self.recommender.users_data.std(), 1))
        self.assertTrue(np.isclose(self.recommender.items_data.mean(), 0))
        self.assertTrue(np.isclose(self.recommender.items_data.std(), 1))

    def test_recommendation_shape(self):
        scores = self.recommender.recommend()
        self.assertEqual(scores.shape, (3, 5))  # 3 users, 5 items

    def test_cosine_similarity(self):
        scores = self.recommender.recommend()
        # For identical vectors, cosine similarity should be 1
        self.assertTrue(np.isclose(scores.diagonal().min(), 1))
        self.assertTrue(np.isclose(scores.diagonal().max(), 1))

    def test_top_N_recommendations(self):
        scores = self.recommender.recommend()
        n = 2
        top_N_items = scores.argsort(axis=1)[:, -n:]
        self.assertEqual(top_N_items.shape, (3, 2))

if __name__ == "__main__":
    unittest.main()
