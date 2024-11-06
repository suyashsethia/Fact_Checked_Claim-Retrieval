import os
import ast
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from data.cleaning import *
# Helper functions
def parse_col(s):
    return ast.literal_eval(s.replace('\n', '\\n')) if s else s

def preprocess_text(text):


    # Apply all transformations
    text = text.lower()
    text = remove_urls(text)
    text = remove_emojis(text)
    text = replace_stops(text)
    text = replace_whitespaces(text)
    text = clean_ocr(text)
    text = clean_twitter_picture_links(text)
    text = clean_twitter_links(text)
    text = remove_elongation(text)
    
    return text

# Load and preprocess data
def load_and_preprocess_data(our_dataset_path):
    posts_path = os.path.join(our_dataset_path, 'posts.csv')
    fact_checks_path = os.path.join(our_dataset_path, 'fact_checks.csv')
    fact_check_post_mapping_path = os.path.join(our_dataset_path, 'pairs.csv')

    for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
        assert os.path.isfile(path)

    # Load CSV files
    df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
    for col in ['claim', 'instances', 'title']:
        df_fact_checks[col] = df_fact_checks[col].apply(parse_col)

    df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
    for col in ['instances', 'ocr', 'verdicts', 'text']:
        df_posts[col] = df_posts[col].apply(parse_col)

    df_fact_check_post_mapping = pd.read_csv(fact_check_post_mapping_path)

    # Load task configurations
    with open('tasks.json') as f:
        data_tasks = json.load(f)['monolingual']

    # Extract posts and fact checks
    fact_checks_ = []
    posts__train = []
    posts__dev = []
    for key, value in data_tasks.items():
        fact_checks_.extend(value['fact_checks'])
        posts__train.extend(value['posts_train'])
        posts__dev.extend(value['posts_dev'])

    df_fact_checks_ = df_fact_checks[df_fact_checks.index.isin(fact_checks_)]
    posts__train, posts__validate = train_test_split(posts__train, test_size=0.2, random_state=42)
    df_posts__train = df_posts[df_posts.index.isin(posts__train)]
    df_posts__validate = df_posts[df_posts.index.isin(posts__validate)]
    df_posts__dev = df_posts[df_posts.index.isin(posts__dev)]

    # Add 'data' column based on text or OCR
    for df in [df_posts__train, df_posts__validate, df_posts__dev]:
        df['text'] = df.apply(lambda x: x['text'][0] if x['text'] != '' else '', axis=1)
        df['ocr'] = df.apply(lambda x: x['ocr'][0][0] if len(x['ocr']) != 0 else '', axis=1)
        df['data'] = df.apply(lambda x: x['text'] if x['text'] != '' else x['ocr'], axis=1)
        df.drop(columns=['instances', 'ocr', 'verdicts', 'text'], inplace=True)
        df.dropna(subset=['data'], inplace=True)

    # Preprocess 'data' and 'claim'
    df_posts__train['data'] = df_posts__train['data'].apply(preprocess_text)
    df_posts__validate['data'] = df_posts__validate['data'].apply(preprocess_text)
    df_posts__dev['data'] = df_posts__dev['data'].apply(preprocess_text)
    df_fact_checks_['claim'] = df_fact_checks_['claim'].apply(lambda x: x[1])
    df_fact_checks_['claim'] = df_fact_checks_['claim'].apply(preprocess_text)

    # Join with fact check mapping
    df_posts__train = df_posts__train.join(df_fact_check_post_mapping.set_index('post_id'), how='inner')
    df_posts__validate = df_posts__validate.join(df_fact_check_post_mapping.set_index('post_id'), how='inner')

    # Merge with fact checks
    df_posts__train = df_posts__train.join(df_fact_checks_, on='fact_check_id', how='inner')
    df_posts__validate = df_posts__validate.join(df_fact_checks_, on='fact_check_id', how='inner')

    # Reset indices
    df_posts__train.reset_index(drop=True, inplace=True)
    df_posts__validate.reset_index(drop=True, inplace=True)

    return df_posts__train, df_posts__validate, df_posts__dev, df_fact_checks_

