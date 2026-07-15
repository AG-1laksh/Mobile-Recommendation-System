"""Feature engineering helpers for the recommendation system."""
import os
import re
import pandas as pd
import numpy as np

def extract_ram_gb(value):
    if pd.isna(value):
        return np.nan

    text = str(value)
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(TB|GB|MB)', text, flags=re.IGNORECASE)

    if not matches:
        return np.nan

    converted_values = []
    for number, unit in matches:
        numeric_value = float(number)
        unit = unit.upper()

        if unit == "TB":
            numeric_value *= 1024
        elif unit == "MB":
            numeric_value /= 1024

        converted_values.append(numeric_value)

    return max(converted_values)

def extract_storage_gb(value):
    if pd.isna(value):
        return np.nan

    text = str(value)
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(TB|GB|MB)', text, flags=re.IGNORECASE)

    if not matches:
        return np.nan

    converted_values = []
    for number, unit in matches:
        numeric_value = float(number)
        unit = unit.upper()

        if unit == "TB":
            numeric_value *= 1024
        elif unit == "MB":
            numeric_value /= 1024

        converted_values.append(numeric_value)

    return max(converted_values)

def extract_battery_mAh(value):
    if pd.isna(value):
        return np.nan

    text = str(value)
    match = re.search(r'(\d+(?:\.\d+)?)\s*mAh', text, flags=re.IGNORECASE)

    if match:
        return float(match.group(1))

    return np.nan

def extract_screen_size_inch(value):
    if pd.isna(value):
        return np.nan

    text = str(value)

    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:inch|inches|in|\")',
        r'(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:\u201d|\u201c)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    return np.nan

def extract_refresh_rate_hz(value):
    if pd.isna(value):
        return 60.0

    text = str(value)
    match = re.search(r'(\d+)\s*Hz', text, flags=re.IGNORECASE)

    if match:
        return float(match.group(1))

    return 60.0

def normalize(series):
    numeric_series = pd.to_numeric(series, errors="coerce")
    normalized_series = pd.Series(5.0, index=series.index, dtype=float)

    valid_values = numeric_series.dropna()
    if valid_values.empty:
        return normalized_series

    minimum_value = valid_values.min()
    maximum_value = valid_values.max()

    if minimum_value == maximum_value:
        return normalized_series

    normalized_values = 10 * (numeric_series - minimum_value) / (maximum_value - minimum_value)
    normalized_values = normalized_values.clip(0, 10)
    normalized_values = normalized_values.fillna(5.0)

    return normalized_values.astype(float)

def encode_ois_flag(value):
    if pd.isna(value):
        return 0

    text = str(value).strip().lower()
    return int(text in {"yes", "true", "1", "y", "ois"})

def encode_ai_score(value):
    if pd.isna(value):
        return 0.0

    text = str(value).strip().lower()

    if "galaxy ai" in text:
        return 10.0
    if "basic ai" in text:
        return 5.0
    if text in {"no", "none", "no ai", "0", "", "na", "n/a"}:
        return 0.0

    return 0.0

def run_feature_engineering(input_path, output_path):
    print(f"Loading cleaned dataset from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    df = pd.read_csv(input_path)

    # 1) Parse raw numeric features
    df["RAM_GB"] = df["RAM"].apply(extract_ram_gb)
    df["Storage_GB"] = df["Storage"].apply(extract_storage_gb)
    df["Battery_mAh"] = df["Battery"].apply(extract_battery_mAh)
    df["Screen_Size_Inch"] = df["Display"].apply(extract_screen_size_inch)
    df["Refresh_Rate_Hz"] = df["Display"].apply(extract_refresh_rate_hz)

    # 2) Impute missing values with medians (or sensible defaults)
    df["RAM_GB"] = df["RAM_GB"].fillna(df["RAM_GB"].median() if not df["RAM_GB"].isna().all() else 6.0)
    df["Storage_GB"] = df["Storage_GB"].fillna(df["Storage_GB"].median() if not df["Storage_GB"].isna().all() else 128.0)
    df["Battery_mAh"] = df["Battery_mAh"].fillna(df["Battery_mAh"].median() if not df["Battery_mAh"].isna().all() else 4500.0)
    df["Screen_Size_Inch"] = df["Screen_Size_Inch"].fillna(df["Screen_Size_Inch"].median() if not df["Screen_Size_Inch"].isna().all() else 6.5)
    df["Refresh_Rate_Hz"] = df["Refresh_Rate_Hz"].fillna(60.0)

    # 3) Compute scores
    df["camera_score"] = (
        0.60 * normalize(df["Main_Camera_MP"]) +
        0.20 * normalize(df["UltraWide_MP"]) +
        0.15 * normalize(df["Telephoto_MP"]) +
        0.05 * (df["OIS"].apply(encode_ois_flag) * 10)
    )

    df["performance_score"] = (
        0.70 * normalize(df["RAM_GB"]) +
        0.30 * normalize(df["Storage_GB"])
    )

    df["battery_score"] = normalize(df["Battery_mAh"])

    df["display_score"] = (
        0.50 * normalize(df["Screen_Size_Inch"]) +
        0.50 * normalize(df["Refresh_Rate_Hz"])
    )

    df["ai_score"] = df["AI_Features"].apply(encode_ai_score)
    
    df["durability_score"] = df["Waterproof"].apply(
        lambda value: 10.0 if str(value).strip().lower() in {"yes", "true", "1", "y"} else 0.0
    )

    # Final raw recommendation score (weights from notebook cell 9)
    recommendation_score_raw = (
        0.30 * df["performance_score"] +
        0.25 * df["camera_score"] +
        0.15 * df["battery_score"] +
        0.15 * df["display_score"] +
        0.10 * df["ai_score"] +
        0.05 * df["durability_score"]
    )

    # Normalize final score
    df["recommendation_score"] = normalize(recommendation_score_raw)

    # Save engineered dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved dataset to: {output_path} with shape: {df.shape}")
    return df

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "../data/processed/cleaned_dataset.csv")
    output_file = os.path.join(script_dir, "../data/processed/engineered_dataset.csv")
    run_feature_engineering(input_file, output_file)
