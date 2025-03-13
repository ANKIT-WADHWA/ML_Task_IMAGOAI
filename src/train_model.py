import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
def load_data():
    X_scaled = np.load("models/X_scaled.npy")  # Load scaled features
    y = np.load("models/y_scaled.npy")        # Load original y, not scaled
    scaler_X = joblib.load("models/scaler_X.pkl")
    return X_scaled, y, scaler_X

# Train CatBoostRegressor with GridSearchCV
def train_model():
    # Load data
    X_scaled, y, scaler_X = load_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # CatBoostRegressor Grid Search
    param_grid_cat = {
        'iterations': [3, 7, 15, 20],
        'learning_rate': [1, 0.01, 0.001],
        'depth': [2, 3, 5, 7]
    }

    model_cat = CatBoostRegressor(loss_function='RMSE', verbose=0)

    grid_search_cat = GridSearchCV(model_cat, param_grid_cat, cv=5, scoring='neg_mean_absolute_error', verbose=1)
    grid_search_cat.fit(X_train, y_train)

    # Best model
    best_cat = grid_search_cat.best_estimator_

    # Evaluate on test set
    preds = best_cat.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))  # Calculate RMSE
    r2 = r2_score(y_test, preds)

    # Print metrics
    print(f"📊 Test MAE: {mae:.4f}")
    print(f"📊 Test RMSE: {rmse:.4f}")
    print(f"📊 R² Score: {r2:.4f}")

    # Save the best model
    joblib.dump(best_cat, "models/best_catboost_model.pkl")

    # Visualize actual vs. predicted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.6, label='Predicted', color='blue')  # Predicted values in blue
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')  # Diagonal line for perfect prediction
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')

    # Add legend
    plt.legend()

    plt.show()

    return best_cat, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_model()


# Model Performance Summary
# ✔ Strengths:
# - High R² Score (0.9777) → Your model explains most of the variance.
# - Low MAE & RMSE → Minimal prediction errors.
# - PCA Variance Explained: 96.01% → Effective dimensionality reduction.
