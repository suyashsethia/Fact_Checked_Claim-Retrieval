import torch
import torch.nn as nn
from transformers import AutoModel

class DualEncoder(nn.Module):
    def __init__(self, query_model_name, doc_model_name=None):
        super(DualEncoder, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.doc_encoder = AutoModel.from_pretrained(doc_model_name or query_model_name)

    def encode_query(self, query_inputs):
        query_embeddings = self.query_encoder(**query_inputs).last_hidden_state[:, 0, :]
        return nn.functional.normalize(query_embeddings, p=2, dim=1)

    def encode_document(self, doc_inputs):
        doc_embeddings = self.doc_encoder(**doc_inputs).last_hidden_state[:, 0, :]
        return nn.functional.normalize(doc_embeddings, p=2, dim=1)

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = nn.functional.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = nn.functional.cosine_similarity(anchor, negative, dim=-1)
        loss = torch.relu(self.margin + neg_sim - pos_sim).mean()
        return loss

class HyperbolicTripletLoss(nn.Module):
    def __init__(self, margin=1.0, eps=1e-6):
        super(HyperbolicTripletLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def hyperbolic_distance(self, x, y):
        x_norm = torch.sqrt(torch.sum(x ** 2, dim=-1) + self.eps)
        y_norm = torch.sqrt(torch.sum(y ** 2, dim=-1) + self.eps)
        numerator = torch.sum((x - y) ** 2, dim=-1)
        denominator = (1 - x_norm ** 2) * (1 - y_norm ** 2)
        return torch.acosh(1 + 2 * numerator / denominator)

    def forward(self, anchor, positive, negative):
        pos_dist = self.hyperbolic_distance(anchor, positive)
        neg_dist = self.hyperbolic_distance(anchor, negative)
        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()
        return loss
