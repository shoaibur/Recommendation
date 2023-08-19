import torch
import torch.nn as nn
import torch.optim as optim

from models.deep_learning_based_recommendation import DeepLearningRec
from models.matrix_factorization_based_recommendation import MatrixFactorization


class HybridRecommendation:

    def __init__(self, users_data, items_data, interaction_data, num_epochs=1000, lr=0.01, hidden_size=16, seed=42):

        # Initialize deep learning recommendation system
        self.deep_learning_rec = DeepLearningRec(users_data, items_data, interaction_data, num_epochs, lr, hidden_size, seed)

        # Initialize matrix factorization recommendation system
        self.matrix_factorization_rec = MatrixFactorization(interaction_data, seed)

        # Make alpha a learnable parameter
        self.alpha = nn.Parameter(torch.tensor([0.5]))

        # Create a combined optimizer
        self.optimizer = optim.Adam([self.alpha,
                                     *self.deep_learning_rec.dense1.parameters(),
                                     *self.deep_learning_rec.dense2.parameters(),
                                     self.matrix_factorization_rec.user_factors,
                                     self.matrix_factorization_rec.item_factors], lr=lr)

        self.num_epochs = num_epochs
        self.known_interactions_mask = torch.tensor(interaction_data, dtype=torch.float32) != -1
        self.interaction_data = torch.tensor(interaction_data, dtype=torch.float32)
        self.loss_function = nn.MSELoss(reduction='none')

    def train(self):
        for epoch in range(self.num_epochs):

            # Get predictions from both models
            deep_learning_scores = self.deep_learning_rec.predict()
            matrix_factorization_scores = self.matrix_factorization_rec.predict()

            # Compute the combined scores
            combined_scores = self.alpha * deep_learning_scores + (1 - self.alpha) * matrix_factorization_scores

            # Compute the loss for known interactions only
            losses = self.loss_function(combined_scores, self.interaction_data)
            masked_losses = losses * self.known_interactions_mask.float()
            loss = masked_losses.sum() / self.known_interactions_mask.float().sum()

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self):
        deep_learning_scores = self.deep_learning_rec.predict()
        matrix_factorization_scores = self.matrix_factorization_rec.predict()

        # Combine the scores
        combined_scores = self.alpha * deep_learning_scores + (1 - self.alpha) * matrix_factorization_scores
        return combined_scores

    def get_top_k_items(self, k=2):
      """
      Get top k items for each user.

      Returns:
      - A tensor of shape (num_users, k) containing the indices of top k items for each user.
      """
      scores = self.predict()
      # We want to consider only items that weren't interacted with (mask value of -1) for recommendation.
      # So we'll set the score of already interacted items to a large negative value.
      scores[self.known_interactions_mask] = -float('inf')

      # Get the top k scores indices
      _, top_k_indices = torch.topk(scores, k, dim=1)
      return top_k_indices
