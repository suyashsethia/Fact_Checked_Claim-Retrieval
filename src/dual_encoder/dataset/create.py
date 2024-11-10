import json
import random
import pandas as pd

def load_results_mp_net(file_path):
    with open(file_path) as f:
        return json.load(f)

def get_negative_samples(df_fact_checks_, post_id, results_mp_net, correct_fact_check_id, num_top_negatives=30, num_random_negatives=30):
    mp_net_neg = results_mp_net.get(str(post_id), [])
    if correct_fact_check_id in mp_net_neg:
        mp_net_neg.remove(correct_fact_check_id)

    top_negative_samples = mp_net_neg[:num_top_negatives]
    df_fact_checks_filtered = df_fact_checks_.loc[df_fact_checks_.index != correct_fact_check_id]
    random_negative_samples = random.sample(df_fact_checks_filtered.index.tolist(), num_random_negatives)

    all_negative_samples = top_negative_samples + random_negative_samples
    negative_samples = {neg_id: df_fact_checks_.loc[neg_id, 'claim'] for neg_id in all_negative_samples}
    return negative_samples

def prepare_training_data(df_posts__train, df_fact_checks_, results_mp_net):
    training_data = {}
    for _, post in df_posts__train.iterrows():
        post_id = post['post_id']
        correct_fact_check_id = post['fact_check_id']
        correct_fact_check_data = post['claim']
        negative_samples = get_negative_samples(df_fact_checks_, post_id, results_mp_net, correct_fact_check_id)
        training_data[post_id] = {
            "post_data": post['data'],
            "correct_fact": correct_fact_check_data,
            "negative_samples": negative_samples
        }
    return training_data
