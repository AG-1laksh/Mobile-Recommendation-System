"""Script to populate ChromaDB with the engineered smartphone dataset."""
import os
import pandas as pd
from utils import get_chroma_client, get_or_create_collection, populate_chromadb

def run_db_population(dataset_path, persist_dir):
    print(f"Reading engineered dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Engineered dataset not found at {dataset_path}. Please run feature_engineering.py first.")
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded with {len(df)} records.")
    
    print(f"Initializing ChromaDB in directory: {persist_dir}")
    client = get_chroma_client(persist_dir)
    
    # Get collection
    collection = get_or_create_collection(client)
    
    # Check if collection has documents already, and clear if so to avoid duplicates
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection already has {existing_count} documents. Re-creating collection to refresh data...")
        client.delete_collection("samsung_phones")
        collection = get_or_create_collection(client)
        
    # Populate collection
    populate_chromadb(df, collection)
    print(f"ChromaDB populated! Collection document count: {collection.count()}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(script_dir, "../data/processed/engineered_dataset.csv")
    chroma_directory = os.path.join(script_dir, "../data/chroma_db")
    
    run_db_population(dataset_file, chroma_directory)
