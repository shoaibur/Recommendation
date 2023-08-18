import torch
import torch.nn as nn
import torch.optim as optim

class MatrixFactorization:

    def __init__(self, interaction_data, seed=42, num_factors=3, num_epochs=1000,
                 loss_function=nn.MSELoss(reduction='none'), optimizer_class=optim.SGD, lr=0.01):

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Check for GPU availability and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert interaction data to tensor
        self.interaction_matrix = torch.tensor(interaction_data, dtype=torch.float32).to(self.device)

        # Mask for known interactions
        self.known_interactions_mask = (self.interaction_matrix != -1)

        # Parameters
        self.num_factors = num_factors
        self.num_epochs = num_epochs
        self.loss_function = loss_function

        # Initialize user and item factors using the dimensions of the interaction_data list
        num_users, num_items = len(interaction_data), len(interaction_data[0])
        self.user_factors = torch.randn(num_users, num_factors, requires_grad=True, device=self.device)
        self.item_factors = torch.randn(num_items, num_factors, requires_grad=True, device=self.device)

        # Optimizer
        self.optimizer = optimizer_class([self.user_factors, self.item_factors], lr=lr)

    def train(self):
        for epoch in range(self.num_epochs):
            # Compute predicted scores
            predicted_scores = self.user_factors @ self.item_factors.T

            # Compute the loss
            losses = self.loss_function(predicted_scores, self.interaction_matrix)
            masked_losses = losses * self.known_interactions_mask.float() # Apply the mask
            loss = masked_losses.sum() / self.known_interactions_mask.float().sum() # Average only over known interactions

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self):
        # Return interaction scores using learned factors
        return self.user_factors @ self.item_factors.T

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
