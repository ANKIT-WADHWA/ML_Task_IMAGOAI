from src.preprocess import preprocess_data
from src.dimensionality_reduction import apply_pca, apply_tsne, visualize_reduction
from src.train_model import train_model
import joblib

file_path = r"D:\ML_Task_ImagoAI\notebooks\data\cleaned_data.csv"

print("\n📌 Preprocessing Data...")
X_scaled, y, scaler_X = preprocess_data(file_path)

print("\n📌 Applying PCA & t-SNE for Visualization...")
X_pca = apply_pca(X_scaled)
X_tsne = apply_tsne(X_scaled)  # Apply t-SNE on scaled features
X_umap = None  # Set to None if not using UMAP

# Visualize results
visualize_reduction(X_pca, X_tsne, X_umap, y)

print("\n🚀 Model Training & Evaluation...")
best_cat, X_train, X_test, y_train, y_test = train_model()

print("\n🎯 Task Completed Successfully!")# Model Performance Summary




# ✔ Strengths:

# High R² Score (0.9777) → Your model explains most of the variance.
# Low MAE & RMSE → Minimal prediction errors.
# PCA Variance Explained: 96.01% → Effective dimensionality reduction.