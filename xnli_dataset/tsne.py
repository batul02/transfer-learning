import joblib
import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Assume these are already available:
# X_source_scaled: Scaled source features (n_source, d)
# X_target_scaled: Scaled target features (n_target, d)
X_source_scaled = np.load("../X_source_scaled.npy")
X_target_scaled = np.load("../X_target_scaled.npy")

# Optional: Subsample for fast t-SNE
n_samples = 1000
X_source_sub = X_source_scaled[:n_samples]
X_target_sub = X_target_scaled[:n_samples]

# ------------------- t-SNE BEFORE TRANSFER -------------------
X_combined_before = np.vstack([X_source_sub, X_target_sub])
labels_before = np.array(['Source'] * len(X_source_sub) + ['Target'] * len(X_target_sub))

tsne_before = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded_before = tsne_before.fit_transform(X_combined_before)

plt.figure(figsize=(8, 6))
for label in ['Source', 'Target']:
    idx = labels_before == label
    plt.scatter(X_embedded_before[idx, 0], X_embedded_before[idx, 1], label=label, alpha=0.6)
plt.title("t-SNE Before Transfer")
plt.legend()
plt.tight_layout()
plt.savefig("tsne_before_projection.png")
plt.show()


# ------------------- LEARN PROJECTION (Ridge) -------------------
# Learn Ŵ such that W * X_source ≈ X_target
# For this we need correspondence data: X_src_corr, X_tgt_corr
# You must already have that computed, e.g., via SHAP, LIME, RF

# Example: assume paired correspondence samples
# X_src_corr.shape = (n_corr, d), X_tgt_corr.shape = (n_corr, d)

corresponding_en, corresponding_fr = joblib.load("explainable_correspondences_shap_rf.pkl")

# Check the shape
print("Source correspondence shape:", corresponding_en.shape)
print("Target correspondence shape:", corresponding_fr.shape)

ridge = Ridge(alpha=1e-5)
ridge.fit(corresponding_en, corresponding_fr)  # Learn transformation
P_matrix = ridge.coef_
X_source_transformed = X_source_scaled @ P_matrix.T


# ------------------- t-SNE AFTER TRANSFER -------------------
X_source_trans_sub = X_source_transformed[:n_samples]
X_combined_after = np.vstack([X_source_trans_sub, X_target_sub])
labels_after = np.array(['Transformed Source'] * len(X_source_trans_sub) + ['Target'] * len(X_target_sub))

tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded_after = tsne_after.fit_transform(X_combined_after)

plt.figure(figsize=(8, 6))
for label in ['Transformed Source', 'Target']:
    idx = labels_after == label
    plt.scatter(X_embedded_after[idx, 0], X_embedded_after[idx, 1], label=label, alpha=0.6)
plt.title("t-SNE After Transfer (Transformed Source + Target)")
plt.legend()
plt.tight_layout()
plt.savefig("tsne_after_projection.png")
plt.show()
