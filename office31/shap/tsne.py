from sklearn.manifold import TSNE
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import joblib
from collections import Counter
from sklearn.linear_model import Ridge
import joblib
import matplotlib.pyplot as plt

data = torch.load("../extracted_features.pt")
X_amazon = data["X_amazon_feats"]
y_amazon = data["y_amazon"]
X_dslr = data["X_dslr_feats"]
y_dslr = data["y_dslr"]

y_amazon_np = y_amazon.numpy() if isinstance(y_amazon, torch.Tensor) else np.array(y_amazon)
y_dslr_np = y_dslr.numpy() if isinstance(y_dslr, torch.Tensor) else np.array(y_dslr)

# Shapes
print("Amazon features shape:", X_amazon.shape)
print("Amazon labels shape:", y_amazon.shape)
print("DSLR features shape:", X_dslr.shape)
print("DSLR labels shape:", y_dslr.shape)

# Unique labels
print("Amazon unique labels:", np.unique(y_amazon_np))
print("DSLR unique labels:", np.unique(y_dslr_np))

# Samples per label
print("Amazon label distribution:", dict(Counter(y_amazon_np)))
print("DSLR label distribution:", dict(Counter(y_dslr_np)))

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_amazon)
X_target_scaled = scaler.transform(X_dslr)

y_source = y_amazon
y_target = y_dslr

X_source_sub = X_source_scaled
X_target_sub = X_target_scaled

# ------------------- t-SNE BEFORE TRANSFER -------------------
# X_combined_before = np.vstack([X_source_sub, X_target_sub])
# labels_before = np.array(['Amazon'] * len(X_source_sub) + ['DSLR'] * len(X_target_sub))

# tsne_before = TSNE(n_components=2, random_state=42, perplexity=30)
# X_embedded_before = tsne_before.fit_transform(X_combined_before)

# plt.figure(figsize=(8, 6))
# for label in ['Amazon', 'DSLR']:
#     idx = labels_before == label
#     plt.scatter(X_embedded_before[idx, 0], X_embedded_before[idx, 1], label=label, alpha=0.6)
# plt.title("t-SNE Before Transfer")
# plt.legend()
# plt.tight_layout()
# plt.savefig("tsne_before_projection.png")
# plt.show()

X_combined_before = np.vstack([X_source_sub, X_target_sub])
y_combined = np.hstack([y_source, y_target])
domain_labels = np.array(['Source'] * len(X_source_sub) + ['Target'] * len(X_target_sub))

# t-SNE
tsne_before = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded_before = tsne_before.fit_transform(X_combined_before)

# Plot class-wise per domain
plt.figure(figsize=(10, 8))
unique_classes = np.unique(y_combined)

for cls in unique_classes:
    # Source points of this class
    src_mask = (domain_labels == 'Source') & (y_combined == cls)
    tgt_mask = (domain_labels == 'Target') & (y_combined == cls)
    
    plt.scatter(X_embedded_before[src_mask, 0], X_embedded_before[src_mask, 1],
                label=f"Source - Class {cls}", alpha=0.6, marker='o')
    plt.scatter(X_embedded_before[tgt_mask, 0], X_embedded_before[tgt_mask, 1],
                label=f"Target - Class {cls}", alpha=0.6, marker='^')

plt.title("t-SNE After Transfer (Class-wise)")
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig("tsne_before_projection_classwise.png")
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
X_source_trans_sub = X_source_transformed
# X_combined_after = np.vstack([X_source_trans_sub, X_target_sub])
# labels_after = np.array(['Transformed Amazon'] * len(X_source_trans_sub) + ['DSLR'] * len(X_target_sub))

# tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)
# X_embedded_after = tsne_after.fit_transform(X_combined_after)

# plt.figure(figsize=(8, 6))
# for label in ['Transformed Amazon', 'DSLR']:
#     idx = labels_after == label
#     plt.scatter(X_embedded_after[idx, 0], X_embedded_after[idx, 1], label=label, alpha=0.6)
# plt.title("t-SNE After Transfer (Transformed Amazon + DSLR)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("tsne_after_projection.png")
# plt.show()

X_combined_after = np.vstack([X_source_trans_sub, X_target_sub])
domain_labels_after = np.array(['Transformed Source'] * len(X_source_trans_sub) + ['Target'] * len(X_target_sub))
y_combined_after = np.hstack([y_source,y_target])

# t-SNE
tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded_after = tsne_after.fit_transform(X_combined_after)

# Plot class-wise per domain
plt.figure(figsize=(10, 8))
for cls in unique_classes:
    src_mask = (domain_labels_after == 'Transformed Source') & (y_combined_after == cls)
    tgt_mask = (domain_labels_after == 'Target') & (y_combined_after == cls)
    
    plt.scatter(X_embedded_after[src_mask, 0], X_embedded_after[src_mask, 1],
                label=f"Transformed Source - Class {cls}", alpha=0.6, marker='o')
    plt.scatter(X_embedded_after[tgt_mask, 0], X_embedded_after[tgt_mask, 1],
                label=f"Target - Class {cls}", alpha=0.6, marker='^')

plt.title("t-SNE Before Transfer (Class-wise)")
plt.legend(loc='best', fontsize=8)
plt.tight_layout()
plt.savefig("tsne_after_projection_classwise.png")
plt.show()


