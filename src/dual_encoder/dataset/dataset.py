import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random

class TripletDataset(Dataset):
    def __init__(self, training_data, tokenizer, max_len=128):
        self.training_data = training_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        post_id = list(self.training_data.keys())[idx]
        post_data = self.training_data[post_id]["post_data"]
        correct_fact = self.training_data[post_id]["correct_fact"]
        negative_samples = self.training_data[post_id]["negative_samples"]
        negative_doc = random.choice(list(negative_samples.values()))

        query_enc = self.tokenizer(post_data, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        pos_doc_enc = self.tokenizer(correct_fact, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")
        neg_doc_enc = self.tokenizer(negative_doc, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")

        return query_enc, pos_doc_enc, neg_doc_enc
