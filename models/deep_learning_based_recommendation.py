import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DeepLearningRec:
    def __init__(self, users_data, items_data, interaction_data, num_epochs=1000, lr=0.01, hidden_size=16, seed=42):

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.users_data = torch.tensor(users_data)
        self.items_data = torch.tensor(items_data)
        self.interaction_matrix = torch.tensor(interaction_data, dtype=torch.float32)
        self.known_interactions_mask = (self.interaction_matrix != -1)

        # Define the dense layers
        input_size = self.users_data.shape[1] + self.items_data.shape[1]
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, 1)

        # Loss and optimizer
        self.loss_function = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam([*self.dense1.parameters(), *self.dense2.parameters()], lr=lr)

        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            all_predictions = []

            for user_features in self.users_data:
                for item_features in self.items_data:
                    combined = torch.cat((user_features, item_features))
                    dense1_output = F.relu(self.dense1(combined))
                    dense2_output = self.dense2(dense1_output)
                    all_predictions.append(dense2_output)

            all_predictions_tensor = torch.stack(all_predictions).view(self.users_data.shape[0], self.items_data.shape[0])

            # Compute loss for known interactions
            losses = self.loss_function(all_predictions_tensor, self.interaction_matrix)
            masked_losses = losses * self.known_interactions_mask.float()
            loss = masked_losses.sum() / self.known_interactions_mask.float().sum()

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def predict(self):
        final_scores = []
        for user_features in self.users_data:
            for item_features in self.items_data:
                combined = torch.cat((user_features, item_features))
                dense1_output = F.relu(self.dense1(combined))
                dense2_output = self.dense2(dense1_output)
                final_scores.append(dense2_output)
        final_scores_tensor = torch.stack(final_scores).view(self.users_data.shape[0], self.items_data.shape[0])
        return final_scores_tensor

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
