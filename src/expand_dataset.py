"""Script to expand the raw Samsung smartphone dataset with post-2023 models."""
import os
import re
import pandas as pd
import numpy as np

def parse_megapixels_from_camera(camera_str):
    """Parses main camera megapixels from raw camera spec string."""
    if pd.isna(camera_str):
        return 8 # Default fallback
        
    text = str(camera_str).strip()
    
    # 1) Search for resolution like 3264 x 2448
    res_match = re.findall(r'(\d+)\s*x\s*(\d+)', text)
    if res_match:
        width, height = int(res_match[0][0]), int(res_match[0][1])
        mp = round((width * height) / 1000000.0)
        return max(mp, 1) # Ensure at least 1 MP
        
    # 2) Search for explicit MP like '12 MP'
    mp_match = re.search(r'(\d+(?:\.\d+)?)\s*MP', text, re.IGNORECASE)
    if mp_match:
        return int(float(mp_match.group(1)))
        
    return 8 # Default fallback

def parse_series_from_name(name):
    """Classifies device into series based on its name."""
    name_upper = str(name).upper()
    if "GALAXY S" in name_upper:
        return "Galaxy S Series"
    elif "GALAXY A" in name_upper:
        return "Galaxy A Series"
    elif "GALAXY M" in name_upper:
        return "Galaxy M Series"
    elif "GALAXY F" in name_upper:
        return "Galaxy F Series"
    elif "GALAXY Z" in name_upper:
        return "Galaxy Z Series"
    elif "XCOVER" in name_upper or "ACTIVE" in name_upper:
        return "Galaxy XCover Series"
    elif "TAB" in name_upper:
        return "Galaxy Tab Series"
    else:
        return "Other"

def parse_segment_from_name(name):
    """Infers segment category for older models based on keywords."""
    name_upper = str(name).upper()
    if any(x in name_upper for x in ["ULTRA", "FOLD", "FLIP", "EDGE", "PRO"]):
        return "Flagship"
    elif "XCOVER" in name_upper or "ACTIVE" in name_upper:
        return "Rugged"
    elif "ZOOM" in name_upper or "CAMERA" in name_upper:
        return "Photography"
    elif "TAB" in name_upper:
        return "Budget"
    else:
        return "Budget"

def expand_dataset(raw_path, interim_path):
    print(f"Loading raw dataset from: {raw_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        
    df_raw = pd.read_csv(raw_path)
    print(f"Loaded raw dataset with shape: {df_raw.shape}")
    
    # 1) Expand columns of older devices
    print("Expanding columns for older devices...")
    df_older = df_raw.copy()
    
    # Launch_Year, Launch_Price, and Current_Price are already in df_raw.
    df_older["Series"] = df_older["Name"].apply(parse_series_from_name)
    df_older["Main_Camera_MP"] = df_older["Camera"].apply(parse_megapixels_from_camera)
    df_older["UltraWide_MP"] = np.nan
    df_older["Telephoto_MP"] = np.nan
    df_older["Front_Camera_MP"] = np.nan
    df_older["OIS"] = "No"
    df_older["AI_Features"] = "No"
    df_older["Waterproof"] = "No"
    df_older["Target_Segment"] = df_older["Name"].apply(parse_segment_from_name)
    
    # Ensure correct columns order
    columns_order = [
        "Name", "Dimensions", "SoC", "CPU", "GPU", "RAM", "Storage", "Display", "Battery", "OS", "Camera",
        "Launch_Year", "Launch_Price", "Current_Price", "Series", "Main_Camera_MP", "UltraWide_MP", "Telephoto_MP", "Front_Camera_MP",
        "OIS", "AI_Features", "Waterproof", "Target_Segment"
    ]
    df_older = df_older[columns_order]
    
    # 2) Define post-2023 Samsung phones and tablets (29 phones + 3 tablets)
    print("Defining post-2023 devices with pricing...")
    new_devices = [
        # Phones
        {"Name": "Galaxy A15 5G", "SoC": "MediaTek Dimensity 6100+", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 17999, "Current_Price": 14999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 5.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "No", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy A16 5G", "SoC": "MediaTek Dimensity 6300", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 18999, "Current_Price": 16999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 5.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "No", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy A25 5G", "SoC": "Exynos 1280", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 26999, "Current_Price": 21999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "Yes", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy A26 5G", "SoC": "Exynos 1330", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 28999, "Current_Price": 24999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "Yes", "Target_Segment": "Budget"},
        {"Name": "Galaxy A35 5G", "SoC": "Exynos 1380", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 30999, "Current_Price": 26999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "Yes", "Target_Segment": "Budget"},
        {"Name": "Galaxy A36 5G", "SoC": "Snapdragon 6 Gen 3", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 32999, "Current_Price": 29999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 16.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "Yes", "Target_Segment": "Budget"},
        {"Name": "Galaxy A55 5G", "SoC": "Exynos 1480", "RAM": "12 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 39999, "Current_Price": 34999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 32.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "Yes", "Target_Segment": "Photography"},
        {"Name": "Galaxy A56 5G", "SoC": "Exynos 1580", "RAM": "12 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 42999, "Current_Price": 39999, "Series": "Galaxy A Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 32.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Photography"},
        {"Name": "Galaxy M15 5G", "SoC": "MediaTek Dimensity 6100+", "RAM": "8 GB", "Storage": "256 GB", "Battery": "6000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 13999, "Current_Price": 11999, "Series": "Galaxy M Series", "Main_Camera_MP": 50, "UltraWide_MP": 5.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "No", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy M35 5G", "SoC": "Exynos 1380", "RAM": "8 GB", "Storage": "256 GB", "Battery": "6000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 19999, "Current_Price": 16999, "Series": "Galaxy M Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "Yes", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy M36 5G", "SoC": "Exynos 1480", "RAM": "8 GB", "Storage": "256 GB", "Battery": "6000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 22999, "Current_Price": 19999, "Series": "Galaxy M Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 16.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy M56 5G", "SoC": "Exynos 1580", "RAM": "12 GB", "Storage": "256 GB", "Battery": "6000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 29999, "Current_Price": 26999, "Series": "Galaxy M Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 32.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "No", "Target_Segment": "Photography"},
        {"Name": "Galaxy F15 5G", "SoC": "MediaTek Dimensity 6100+", "RAM": "8 GB", "Storage": "256 GB", "Battery": "6000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 12999, "Current_Price": 10999, "Series": "Galaxy F Series", "Main_Camera_MP": 50, "UltraWide_MP": 5.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "No", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy F35 5G", "SoC": "Exynos 1380", "RAM": "8 GB", "Storage": "256 GB", "Battery": "6000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 19999, "Current_Price": 16999, "Series": "Galaxy F Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "Yes", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"},
        {"Name": "Galaxy F55 5G", "SoC": "Snapdragon 7 Gen 1", "RAM": "12 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 26999, "Current_Price": 22999, "Series": "Galaxy F Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 50.0, "OIS": "Yes", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Photography"},
        {"Name": "Galaxy F56 5G", "SoC": "Snapdragon 7s Gen 2", "RAM": "12 GB", "Storage": "256 GB", "Battery": "5000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 27999, "Current_Price": 24999, "Series": "Galaxy F Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 50.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "No", "Target_Segment": "Photography"},
        {"Name": "Galaxy S24 FE", "SoC": "Exynos 2400e", "RAM": "8 GB", "Storage": "256 GB", "Battery": "4700 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 59999, "Current_Price": 49999, "Series": "Galaxy S Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": 8.0, "Front_Camera_MP": 10.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy S24", "SoC": "Snapdragon 8 Gen 3 / Exynos 2400", "RAM": "8 GB", "Storage": "256 GB", "Battery": "4000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 79999, "Current_Price": 64999, "Series": "Galaxy S Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": 10.0, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy S24+", "SoC": "Snapdragon 8 Gen 3 / Exynos 2400", "RAM": "12 GB", "Storage": "256 GB", "Battery": "4900 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 99999, "Current_Price": 84999, "Series": "Galaxy S Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": 10.0, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy S24 Ultra", "SoC": "Snapdragon 8 Gen 3 for Galaxy", "RAM": "12 GB", "Storage": "512 GB", "Battery": "5000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 129999, "Current_Price": 109999, "Series": "Galaxy S Series", "Main_Camera_MP": 200, "UltraWide_MP": 12.0, "Telephoto_MP": 50.0, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy S25", "SoC": "Snapdragon 8 Elite", "RAM": "12 GB", "Storage": "256 GB", "Battery": "4000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 84999, "Current_Price": 79999, "Series": "Galaxy S Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": 10.0, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy S25+", "SoC": "Snapdragon 8 Elite", "RAM": "12 GB", "Storage": "256 GB", "Battery": "4900 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 104999, "Current_Price": 99999, "Series": "Galaxy S Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": 10.0, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy S25 Edge", "SoC": "Snapdragon 8 Elite", "RAM": "12 GB", "Storage": "256 GB", "Battery": "4900 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 104999, "Current_Price": 99999, "Series": "Galaxy S Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": 10.0, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy S25 Ultra", "SoC": "Snapdragon 8 Elite for Galaxy", "RAM": "16 GB", "Storage": "512 GB", "Battery": "5000 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 144999, "Current_Price": 139999, "Series": "Galaxy S Series", "Main_Camera_MP": 200, "UltraWide_MP": 50.0, "Telephoto_MP": 50.0, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy Z Flip6", "SoC": "Snapdragon 8 Gen 3 for Galaxy", "RAM": "12 GB", "Storage": "256 GB", "Battery": "4000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 109999, "Current_Price": 94999, "Series": "Galaxy Z Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 10.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy Z Fold6", "SoC": "Snapdragon 8 Gen 3 for Galaxy", "RAM": "12 GB", "Storage": "512 GB", "Battery": "4400 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 164999, "Current_Price": 149999, "Series": "Galaxy Z Series", "Main_Camera_MP": 50, "UltraWide_MP": 12.0, "Telephoto_MP": 10.0, "Front_Camera_MP": 10.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy XCover7", "SoC": "MediaTek Dimensity 6100+", "RAM": "6 GB", "Storage": "128 GB", "Battery": "4050 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 29000, "Current_Price": 24999, "Series": "Galaxy XCover Series", "Main_Camera_MP": 50, "UltraWide_MP": np.nan, "Telephoto_MP": np.nan, "Front_Camera_MP": 5.0, "OIS": "No", "AI_Features": "No", "Waterproof": "Yes", "Target_Segment": "Rugged"},
        {"Name": "Galaxy XCover7 Pro", "SoC": "Snapdragon 7s Gen 2", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5050 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 38000, "Current_Price": 32999, "Series": "Galaxy XCover Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "Yes", "AI_Features": "No", "Waterproof": "Yes", "Target_Segment": "Rugged"},
        {"Name": "Galaxy XCover8 Pro", "SoC": "Snapdragon 7 Gen 3", "RAM": "8 GB", "Storage": "256 GB", "Battery": "5050 mAh", "OS": "Android 15", "Launch_Year": 2025.0, "Launch_Price": 45000, "Current_Price": 39999, "Series": "Galaxy XCover Series", "Main_Camera_MP": 50, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 13.0, "OIS": "Yes", "AI_Features": "Basic AI", "Waterproof": "Yes", "Target_Segment": "Rugged"},
        # Tablets
        {"Name": "Galaxy Tab S9 FE", "SoC": "Exynos 1380", "RAM": "6 GB", "Storage": "128 GB", "Battery": "8000 mAh", "OS": "Android 14", "Launch_Year": 2024.0, "Launch_Price": 36999, "Current_Price": 29999, "Series": "Galaxy Tab Series", "Main_Camera_MP": 8, "UltraWide_MP": 0.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 12.0, "OIS": "No", "AI_Features": "No", "Waterproof": "Yes", "Target_Segment": "Budget"},
        {"Name": "Galaxy Tab S10 Ultra", "SoC": "MediaTek Dimensity 9300+", "RAM": "12 GB", "Storage": "512 GB", "Battery": "11200 mAh", "OS": "Android 15", "Launch_Year": 2024.0, "Launch_Price": 108999, "Current_Price": 99999, "Series": "Galaxy Tab Series", "Main_Camera_MP": 13, "UltraWide_MP": 8.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 12.0, "OIS": "Yes", "AI_Features": "Galaxy AI", "Waterproof": "Yes", "Target_Segment": "Flagship"},
        {"Name": "Galaxy Tab A9+", "SoC": "Snapdragon 695", "RAM": "4 GB", "Storage": "64 GB", "Battery": "7040 mAh", "OS": "Android 13", "Launch_Year": 2024.0, "Launch_Price": 19999, "Current_Price": 16999, "Series": "Galaxy Tab Series", "Main_Camera_MP": 8, "UltraWide_MP": 0.0, "Telephoto_MP": np.nan, "Front_Camera_MP": 5.0, "OIS": "No", "AI_Features": "No", "Waterproof": "No", "Target_Segment": "Budget"}
    ]
    df_new = pd.DataFrame(new_devices)
    
    # Fill missing fields in new devices df with appropriate NaN or placeholders to match old
    for col in columns_order:
        if col not in df_new.columns:
            df_new[col] = np.nan
            
    df_new = df_new[columns_order]
    
    # Concat and save
    df_merged = pd.concat([df_older, df_new], ignore_index=True)
    
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)
    df_merged.to_csv(interim_path, index=False)
    print(f"Dataset expansion completed! Merged dataset saved to: {interim_path} with shape: {df_merged.shape}")
    return df_merged

if __name__ == "__main__":
    # We will rename the paths to Samsung-Phone-Recommendation-System after folder rename,
    # but for now we write the relative path relative to project structure or relative to file location.
    # To be extremely safe, we will use path relative to the script:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_file = os.path.join(script_dir, "../data/raw/SamsungPhoneData.csv")
    interim_file = os.path.join(script_dir, "../data/interim/SamsungPhoneData_Updated.csv")
    expand_dataset(raw_file, interim_file)
