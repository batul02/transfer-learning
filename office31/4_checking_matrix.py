import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch

# Load the saved matrices from the file
data = torch.load("office31_feature_correspondence.pt", weights_only=False)

# Extract the matrices
source_matrix = data["source_matrix"]
target_matrix = data["target_matrix"]

# Now you can check or process the matrices further
print("Source Matrix Shape:", source_matrix.shape)
print("Target Matrix Shape:", target_matrix.shape)


# Check if all values are non-zero
print(np.all(source_matrix == 0))
print(np.all(target_matrix == 0))

# Combine source and target matrices
combined_matrix = np.vstack([source_matrix, target_matrix])
labels = ["Source"] * source_matrix.shape[0] + ["Target"] * target_matrix.shape[0]

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(combined_matrix)

# Plot
plt.figure(figsize=(8, 6))
for label in ["Source", "Target"]:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)

plt.title("PCA of Feature Matrices")
plt.legend()
plt.grid(True)
plt.savefig("PCA_Feature_Matrices.jpeg")

# Feature-level discrepancy
feature_discrepancy = np.abs(source_matrix.mean(axis=0) - target_matrix.mean(axis=0))
mean_discrepancy = feature_discrepancy.mean()
print("Mean Feature Discrepancy:", mean_discrepancy)

# Cosine Similarity (between feature means)
from sklearn.metrics.pairwise import cosine_similarity

source_mean = source_matrix.mean(axis=0).reshape(1, -1)
target_mean = target_matrix.mean(axis=0).reshape(1, -1)

cos_sim = cosine_similarity(source_mean, target_mean)[0][0]
print("Cosine Similarity:", cos_sim)

# Correlation (Pearson)
from scipy.stats import pearsonr

corr, _ = pearsonr(source_mean.flatten(), target_mean.flatten())
print("Pearson Correlation:", corr)

# KL Divergence (after normalizing to distributions)
from scipy.special import rel_entr

# Normalize mean vectors to sum to 1 (like probabilities)
p = source_mean.flatten()
q = target_mean.flatten()
p /= p.sum()
q /= q.sum()

kl_div = np.sum(rel_entr(p, q))
print("KL Divergence (source || target):", kl_div)


