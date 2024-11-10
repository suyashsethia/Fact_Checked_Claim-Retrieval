import torch
import torch.optim as optim
from tqdm import tqdm
from dual_encoder.dataset.model import CosineTripletLoss
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def validation(model, df_fact_checks_, df_post__validation, tokenizer, device='cuda', batch_size=32, top_k=10):
    model.to(device).eval()
    idx_to_fact_check_id = {idx: fact_check_id for idx, fact_check_id in enumerate(df_fact_checks_.index.tolist())}
    fact_check_texts = df_fact_checks_['claim'].tolist()

    fact_check_loader = DataLoader(
        fact_check_texts,
        batch_size=batch_size,
        collate_fn=lambda x: tokenizer(x, padding=True, truncation=True, max_length=128, return_tensors="pt")
    )

    all_facts_embeddings = []
    with torch.no_grad():
        for batch in fact_check_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            embeddings = model.encode_document(batch).cpu()
            all_facts_embeddings.append(embeddings)

    all_facts_embeddings = torch.cat(all_facts_embeddings, dim=0)
    all_facts_embeddings = all_facts_embeddings / torch.norm(all_facts_embeddings, dim=1, keepdim=True)

    correct_retrievals, all_mrr, total_queries = 0, [], 0
    for _, post in tqdm(df_post__validation.iterrows(), total=len(df_post__validation)):
        post_inputs = tokenizer(post['data'], padding=True, truncation=True, max_length=128, return_tensors="pt")
        post_inputs = {key: val.to(device) for key, val in post_inputs.items()}
        with torch.no_grad():
            post_embedding = model.encode_query(post_inputs).cpu()

        post_embedding = post_embedding / torch.norm(post_embedding, dim=1, keepdim=True)
        similarity = torch.matmul(post_embedding, all_facts_embeddings.T).squeeze()

        top_k_indices = similarity.topk(top_k).indices.tolist()
        top_k_fact_ids = [idx_to_fact_check_id[idx] for idx in top_k_indices]
        if post['fact_check_id'] in top_k_fact_ids:
            correct_retrievals += 1
            rank = top_k_fact_ids.index(post['fact_check_id']) + 1
            all_mrr.append(1 / rank)
        else:
            all_mrr.append(0)

        total_queries += 1

    return {
        "accuracy": correct_retrievals / total_queries,
        "mrr": np.mean(all_mrr)
    }

def train_dual_encoder_with_negatives(
    model,
    dataset,
    df_fact_checks_,
    df_posts__validate,
    tokenizer,
    epochs=10,
    batch_size=32,
    lr=2e-5,
    device="cuda",
    top_k=10
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = CosineTripletLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            query_enc, pos_doc_enc, neg_doc_enc = batch
            query_enc = {key: val.squeeze(1).to(device) for key, val in query_enc.items()}
            pos_doc_enc = {key: val.squeeze(1).to(device) for key, val in pos_doc_enc.items()}
            neg_doc_enc = {key: val.squeeze(1).to(device) for key, val in neg_doc_enc.items()}

            query_embeddings = model.encode_query(query_enc)
            pos_embeddings = model.encode_document(pos_doc_enc)
            neg_embeddings = model.encode_document(neg_doc_enc)

            loss = loss_fn(query_embeddings, pos_embeddings, neg_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
