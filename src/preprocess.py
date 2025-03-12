import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Define dataset path
DATA_PATH = r"D:\ML_Task_ImagoAI\notebooks\data\cleaned_data.csv"

def preprocess_data(data_path):
    """
    Load, preprocess, and scale data. Saves scalers and scaled data.
    Returns: hsi_id, X_scaled, y
    """
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Drop hsi_id column (not necessary for modeling)
    df.drop('hsi_id', axis=1, inplace=True)

    # Extract features (X) and target (y)
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column (target variable)

    # Standardize the features (X) only
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save scaler and scaled data
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    np.save("models/X_scaled.npy", X_scaled)
    np.save("models/y_scaled.npy", y)  # Save original y, not scaled

    print("\n✅ Preprocessing Completed!")
    print(f"🔹 Features Shape: {X_scaled.shape}")
    print(f"🔹 Target Shape: {y.shape}")

    return X_scaled, y, scaler_X

if __name__ == "__main__":
    preprocess_data(DATA_PATH)