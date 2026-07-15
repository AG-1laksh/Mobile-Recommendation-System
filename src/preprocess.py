"""Preprocessing utilities for the recommendation system."""
import os
import pandas as pd
import numpy as np

def clean_dataset(input_path, output_path):
    """Loads the updated dataset, standardizes categorical columns and missing values, and saves the cleaned dataset."""
    print(f"Loading dataset from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found at {input_path}. Please place it there first.")
        
    df = pd.read_csv(input_path)
    print(f"Loaded dataset with shape: {df.shape}")

    # Standardize OIS and Waterproof to 'Yes' / 'No' (or clean strings)
    yes_no_columns = ["OIS", "Waterproof"]
    for col in yes_no_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
            )
            print(f"Standardized {col} column.")

    # Standardize AI Features
    if "AI_Features" in df.columns:
        df["AI_Features"] = (
            df["AI_Features"]
            .astype(str)
            .str.strip()
            .replace({
                "None": "No",
                "Nan": "No",
                "nan": "No",
                "NaN": "No"
            })
        )
        print("Standardized AI_Features column.")

    # Handle missing camera megapixels
    camera_columns = ["UltraWide_MP", "Telephoto_MP"]
    for col in camera_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            print(f"Filled missing values in {col} with 0.")

    # Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path} with shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Default paths for execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "../data/interim/SamsungPhoneData_Updated.csv")
    output_file = os.path.join(script_dir, "../data/processed/cleaned_dataset.csv")
    
    # Run the cleaning pipeline
    clean_dataset(input_file, output_file)
