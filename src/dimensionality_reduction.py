import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# Load preprocessed data
X_scaled = np.load("models/X_scaled.npy")
y_scaled = np.load("models/y_scaled.npy").flatten()  # Flatten y for plotting

def apply_pca(X, explained_variance_threshold=0.95):
    """
    Apply PCA for dimensionality reduction while preserving 95% variance.
    Saves reduced data and returns transformed features.
    """
    pca = PCA(n_components=explained_variance_threshold)  # Keep components that explain 95% variance
    X_pca = pca.fit_transform(X)
    
    # Calculate explained variance
    total_variance = np.sum(pca.explained_variance_ratio_)
    print(f"🔹 Explained Variance: {total_variance:.4f}")
    print(f"🔹 Number of Components: {pca.n_components_}")

    # Save reduced data
    np.save("models/X_pca.npy", X_pca)
    joblib.dump(pca, "models/pca_model.pkl")  # Save PCA model
    
    return X_pca

def apply_tsne(X, n_components=2, perplexity=30):
    """
    Apply t-SNE for visualization in 2D space.
    Saves reduced data and returns transformed features.
    """
    try:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        np.save("models/X_tsne.npy", X_tsne)
        return X_tsne
    except ValueError as e:
        print(f"⚠️ t-SNE Error: {e}. Try lowering perplexity.")
        return None

def apply_umap(X, n_components=10):
    """
    Apply UMAP for non-linear dimensionality reduction.
    Saves reduced data and returns transformed features.
    """
    umap = UMAP(n_components=n_components, random_state=42)
    X_umap = umap.fit_transform(X)
    np.save("models/X_umap.npy", X_umap)
    return X_umap

def visualize_reduction(X_pca, X_tsne, X_umap, y):
    """
    Plot PCA, t-SNE, and UMAP results for visualization.
    """
    plt.figure(figsize=(18, 5))

    # PCA Scatter Plot
    plt.subplot(1, 3, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.title('PCA Reduction')

    # t-SNE Scatter Plot (if available)
    if X_tsne is not None:
        plt.subplot(1, 3, 2)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='plasma', alpha=0.5)
        plt.colorbar()
        plt.title('t-SNE Reduction')

    # UMAP Scatter Plot (if available)
    if X_umap is not None:
        plt.subplot(1, 3, 3)
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='cividis', alpha=0.5)
        plt.colorbar()
        plt.title('UMAP Reduction')

    plt.show()

# Apply PCA, t-SNE, and UMAP
X_pca = apply_pca(X_scaled)
X_tsne = apply_tsne(X_pca, n_components=2, perplexity=30)  # ✅ Using dynamic perplexity & components
X_umap = apply_umap(X_scaled, n_components=10)  # Reduce to 10 components using UMAP

# Visualize results
visualize_reduction(X_pca, X_tsne, X_umap, y_scaled)