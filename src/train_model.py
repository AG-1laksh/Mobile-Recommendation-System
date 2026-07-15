"""Model training entry point for the recommendation system."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

def train_model(data_path, model_path):
    print(f"Loading engineered dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # Define features and target
    features = [
        "RAM_GB",
        "Storage_GB",
        "Battery_mAh",
        "Screen_Size_Inch",
        "Refresh_Rate_Hz",
        "Main_Camera_MP",
        "UltraWide_MP",
        "Telephoto_MP",
        "Front_Camera_MP",
        "Launch_Year"
    ]
    target = "recommendation_score"
    
    X = df[features]
    y = df[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data into train size: {X_train.shape[0]} and test size: {X_test.shape[0]}")
    
    # Train XGBoost regressor (XGBoost handles missing/NaN values natively)
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        missing=np.nan
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=" * 35)
    print("XGBOOST MODEL EVALUATION METRICS")
    print("=" * 35)
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"R-squared (R2) Score   : {r2:.6f}")
    print("=" * 35)
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved successfully to: {model_path}")
    return model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "../data/processed/engineered_dataset.csv")
    model_output_path = os.path.join(script_dir, "../data/models/xgboost_model.joblib")
    
    train_model(dataset_path, model_output_path)
