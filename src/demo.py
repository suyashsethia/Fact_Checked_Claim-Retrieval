import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import torch.nn as nn

# Define the DualEncoder model
class DualEncoder(nn.Module):
    def __init__(self, query_model_name, doc_model_name=None):
        super(DualEncoder, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(query_model_name)
        self.doc_encoder = AutoModel.from_pretrained(doc_model_name or query_model_name)

    def encode_query(self, query_inputs):
        query_embeddings = self.query_encoder(**query_inputs).last_hidden_state[:, 0, :]
        query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings

    def encode_document(self, doc_inputs):
        doc_embeddings = self.doc_encoder(**doc_inputs).last_hidden_state[:, 0, :]
        doc_embeddings = nn.functional.normalize(doc_embeddings, p=2, dim=1)
        return doc_embeddings

    def forward(self, query_inputs, doc_inputs):
        query_embeddings = self.encode_query(query_inputs)
        doc_embeddings = self.encode_document(doc_inputs)
        return query_embeddings, doc_embeddings

# Prediction function
def predict_with_sentences(model, posts, df_fact_checks_, tokenizer, batch_size=32, device='cuda', top_k=10):
    model.eval()

    idx_to_fact_check_id = {idx: fact_check_id for idx, fact_check_id in enumerate(df_fact_checks_.index.tolist())}
    idx_to_fact_check_sentence = {idx: claim for idx, claim in enumerate(df_fact_checks_['claim'].tolist())}
    fact_check_texts = df_fact_checks_['claim'].tolist()
    # all_facts_embeddings = []

    # fact_check_loader = DataLoader(
    #     fact_check_texts,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=lambda x: tokenizer(x, padding=True, truncation=True, max_length=128, return_tensors="pt")
    # )

    # with torch.no_grad():
    #     for batch in tqdm(fact_check_loader, desc="Encoding Fact-Checks"):
    #         batch = {key: val.to(device) for key, val in batch.items()}
    #         embeddings = model.encode_document(batch).cpu()
    #         all_facts_embeddings.append(embeddings)

    # all_facts_embeddings = torch.cat(all_facts_embeddings, dim=0)
    # all_facts_embeddings = all_facts_embeddings / torch.norm(all_facts_embeddings, dim=1, keepdim=True)
    all_facts_embeddings = torch.load("all_facts_embeddings_mpnet.pt")
    top_k_results = {}
    post_loader = DataLoader(
        posts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tokenizer(x, padding=True, truncation=True, max_length=128, return_tensors="pt")
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(post_loader, desc="Predicting Posts")):
            batch = {key: val.to(device) for key, val in batch.items()}
            post_embeddings = model.encode_query(batch).cpu()
            post_embeddings = post_embeddings / torch.norm(post_embeddings, dim=1, keepdim=True)
            similarities = torch.matmul(post_embeddings, all_facts_embeddings.T)

            for j, similarity_scores in enumerate(similarities):
                post_idx = i * batch_size + j
                top_k_indices = similarity_scores.topk(top_k).indices.tolist()
                top_k_fact_checks = [
                    {
                        "fact_check_id": idx_to_fact_check_id[idx],
                        "sentence": idx_to_fact_check_sentence[idx]
                    }
                    for idx in top_k_indices
                ]
                top_k_results[post_idx] = top_k_fact_checks

    return top_k_results

# Streamlit app
def main():
    st.title("Dual Encoder Fact-Check Retrieval")
    st.write("Enter posts to find the most relevant fact-checks.")

    # Sidebar: Load model and tokenizer
    st.sidebar.header("Configuration")
    model_name =  "sentence-transformers/all-mpnet-base-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DualEncoder(model_name)
    model.to(device)
    model.load_state_dict(torch.load("model_epoch9"))
    # Load fact-check data
    df_fact_checks_ = pd.read_csv("df_fact_checks_.csv")
    # if fact_check_file:
    #     df_fact_checks_ = pd.read_csv(fact_check_file)
    #     st.write("Fact-Check Dataset Loaded:")
    #     st.write(df_fact_checks_.head())

    # Input posts
    posts = st.text_area("Enter posts (one per line):")
    if st.button("Find Fact-Checks"):
        if posts.strip():
            posts_list = posts.split("\n")
            with st.spinner("Retrieving fact-checks..."):
                predictions = predict_with_sentences(
                    model,
                    posts_list,
                    df_fact_checks_,
                    tokenizer,
                    device=device,
                    top_k=5
                )

            st.write("Results:")
            for post_idx, top_k_fact_checks in predictions.items():
                st.write(f"Post {post_idx + 1}:")
                for rank, fact in enumerate(top_k_fact_checks, start=1):
                    st.write(f"  Rank {rank}:")
                    st.write(f"    Fact-check ID: {fact['fact_check_id']}")
                    st.write(f"    Sentence: {fact['sentence']}")
                st.write("-" * 50)
        else:
            st.error("Please upload a fact-check dataset and enter posts.")

if __name__ == "__main__":
    main()
