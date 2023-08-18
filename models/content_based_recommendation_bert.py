import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity

class ContentBasedRecommendation:
    def __init__(self, users_data, items_data, pretrained_model="bert-base-uncased"):
        self.users_data = users_data
        self.items_data = items_data

        # Load pretrained BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model)

    def _get_embedding(self, data):
        inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def recommend(self):
        # Get BERT embeddings for users and items
        user_embeddings = self._get_embedding(self.users_data)
        item_embeddings = self._get_embedding(self.items_data)

        # Compute similarity scores
        scores = torch.zeros(user_embeddings.size(0), item_embeddings.size(0))
        for i, user_emb in enumerate(user_embeddings):
            for j, item_emb in enumerate(item_embeddings):
                scores[i, j] = cosine_similarity(user_emb.unsqueeze(0), item_emb.unsqueeze(0))
        return scores
