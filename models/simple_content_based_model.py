from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class SimpleContentBasedRec:

    def __init__(self, users_data, items_data):
        # Assuming users_data and items_data are numpy arrays or similar structures
        self.users_data = users_data
        self.items_data = items_data

        # Scale or normalize features
        scaler = StandardScaler()
        self.users_data = scaler.fit_transform(self.users_data)
        self.items_data = scaler.fit_transform(self.items_data)

    def recommend(self):
        # Compute similarity scores
        scores = cosine_similarity(self.users_data, self.items_data)
        return scores
