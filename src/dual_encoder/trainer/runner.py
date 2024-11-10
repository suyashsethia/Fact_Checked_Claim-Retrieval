import argparse
from dual_encoder.dataset.create import prepare_training_data, load_results_mp_net
from dataset import TripletDataset
from dual_encoder.dataset.model import DualEncoder
from dual_encoder.trainer.train import train_dual_encoder_with_negatives, validation
from transformers import AutoTokenizer
import json

from loader import load_and_preprocess_data

# Define dataset path
our_dataset_path = '.'

# Load and preprocess data
df_posts__train, df_posts__validate, df_posts__dev, df_fact_checks_ = load_and_preprocess_data(our_dataset_path)

# Now you can use these dataframes in your training and validation workflow



def main(args):
    # Load data
    print("Loading data...")
    results_mp_net = load_results_mp_net("result_mp_net.json")  # Fixed path for results
    training_data = prepare_training_data(df_posts__train, df_fact_checks_, results_mp_net)

    # Initialize tokenizer and model
    print(f"Initializing model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = DualEncoder(query_model_name=args.model_name)

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = TripletDataset(training_data, tokenizer)

    # Train model
    print("Training model...")
    train_dual_encoder_with_negatives(
        model, 
        dataset, 
        df_fact_checks_, 
        df_posts__validate, 
        tokenizer
    )

    # Validate model
    print("Validating model...")
    metrics = validation(model, df_fact_checks_, df_post__validate, tokenizer)
    print("Validation metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate a DualEncoder model with a specific model name.")
    
    # Model-related argument
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Pretrained model name for DualEncoder (e.g., sentence-transformers/distiluse-base-multilingual-cased-v2)."
    )
    
    # Parse arguments
    args = parser.parse_args()
    main(args)
