import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score
from typing import List, Dict

class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        
    def fit(self, df_fact_checks):
        """Fit the BM25 model with fact-check claims."""
        self.fact_check_ids = df_fact_checks['fact_check_id'].tolist()
        tokenized_claims = [claim.split() for claim in df_fact_checks_['claim']]
        self.bm25 = BM25Okapi(tokenized_claims, k1=self.k1, b=self.b)

    def retrieve_top_k(self, post_data: str, k) -> List[int]:
        """Retrieve the top k fact_check_ids for a given post."""
        tokenized_post = post_data.split()
        scores = self.bm25.get_scores(tokenized_post)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return top_k_indices


class BM25Evaluator:
    def __init__(self, retriever: BM25Retriever):
        self.retriever = retriever

    def evaluate_success_at_10(self, df_posts__validate,k) -> Dict[str, float]:
        """Evaluate the success@10 metric and return the average score and top 10 fact_check_ids."""
        success_at_10 = 0
        top_10_results = {}

        for _, row in df_posts__validate.iterrows():
            post_id = row['post_id']
            # correct_fact_id = row['fact_check_id']
            retrieved_fact_ids = self.retriever.retrieve_top_k(row['data'], k=10)

            # Check if the correct fact ID is in the top 10 retrieved results
            if row['fact_check_id'] in retrieved_fact_ids:
                success_at_10 += 1

        #     # Store the retrieved fact IDs for this post
            top_10_results[post_id] = retrieved_fact_ids

        # Calculate the average success@10 score
        avg_success_at_10 = success_at_10 / len(df_posts__validate)
        
        # return {'average_score': avg_success_at_10, 'top_10_results': top_10_results}
        return top_10_results , avg_success_at_10

# bm25_retriever = BM25Retriever(k1=1.2, b=0.75)
# bm25_retriever.fit(df_fact_checks_)
# with open('BM25API.pkl', 'wb') as f:
#     pickle.dump(bm25_retriever, f)
# Initialize retriever with BM25 param