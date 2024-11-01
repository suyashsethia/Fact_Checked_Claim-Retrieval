import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import os
import pickle

class FactCheckRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_file='fact_check_index.faiss', mapping_file='fact_check_id_mapping.pkl'):
        self.model = SentenceTransformer(model_name)
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.index = None
        self.id_to_fact_check = {}

    def create_index(self, claims, fact_check_ids):
        embeddings = self._encode_texts(claims)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.id_to_fact_check = {i: fact_id for i, fact_id in enumerate(fact_check_ids)}
        self._save_index()
        self._save_mapping()

    def load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.mapping_file, 'rb') as f:
                self.id_to_fact_check = pickle.load(f)

    def retrieve_top_k(self, queries, k=10):
        query_embeddings = self._encode_texts(queries)
        _, top_k_indices = self.index.search(query_embeddings, k)
        return [[self.id_to_fact_check[idx] for idx in indices] for indices in top_k_indices]

    def evaluate(self, posts, k=10):
        results = {}
        for _, post in posts.iterrows():
            post_id = post['post_id']
            correct_fact_id = post['fact_check_id']
            top_k_fact_check_ids = self.retrieve_top_k([post['data']], k=k)[0]
            results[post_id] = int(correct_fact_id in top_k_fact_check_ids)
        return np.mean(list(results.values()))

    def _encode_texts(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return np.array(embeddings).astype('float32')

    def _save_index(self):
        faiss.write_index(self.index, self.index_file)

    def _save_mapping(self):
        with open(self.mapping_file, 'wb') as f:
            pickle.dump(self.id_to_fact_check, f)


# def main():
#     retriever = FactCheckRetriever()
    
#     if not os.path.exists(retriever.index_file):
#         # Assuming `df_fact_checks_` contains 'claim' and 'fact_check_id'
#         retriever.create_index(df_fact_checks_['claim'].tolist(), df_fact_checks_['fact_check_id'].tolist())
#     else:
#         retriever.load_index()
    
#     # Assuming `df_posts__validate` contains 'post_id', 'data', and 'fact_check_id'
#     success_at_10 = retriever.evaluate(df_posts__validate, k=10)
#     print("Average Success@10 Score:", success_at_10)

# if __name__ == "__main__":
#     main()
