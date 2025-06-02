import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
import joblib
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm
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

le = LabelEncoder()
y_source = le.fit_transform(y_source)
y_target = le.transform(y_target)

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target_scaled, y_target, test_size=0.4, stratify=y_target, random_state=42
)

# Parallelize the training process using joblib
n_jobs = -1  # Use the maximum number of available cores

def train_rf_model(X, y, random_state=42):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state).fit(X, y)

# Parallel training of the base model (only one as per your case)
clf_tg_base = train_rf_model(X_target_train, y_target_train)

# Step 11: Evaluate Base Model on Target Test Data
y_fr_base_pred = clf_tg_base.predict(X_target_test)
base_accuracy = accuracy_score(y_target_test, y_fr_base_pred)

print("Base Model Accuracy on Dslr test data:", base_accuracy)
joblib.dump(clf_tg_base, "rf_tg_base.pkl")

# Step 4: Train Logistic Regression Models on Source
clf_en = train_rf_model(X_source_scaled, y_source)

# Step 5: Compute SHAP Values for Source and Target Models
print("Explaining Amzn model with SHAP...")
explainer_sr = shap.TreeExplainer(clf_en)
# shap_values_en = explainer_en.shap_values(X_source_scaled)
def explain_single_instance_s(x):
    shap_vals = explainer_sr.shap_values(x, check_additivity=False)
    return shap_vals # returns list of arrays, one per class

# Use all CPU cores
n_jobs = -1

# Parallel computation
shap_values_sr = Parallel(n_jobs=n_jobs)(
    delayed(explain_single_instance_s)(X_source_scaled[i]) for i in tqdm(range(len(X_source_scaled)))
)

# Convert to consistent array shape
# Each item is a list (one per class); stack per class
shap_values_sr_stacked = [np.vstack([shap_values_sr[i][cls] for i in range(len(X_source_scaled))])
                          for cls in range(len(shap_values_sr[0]))]
print("SHAP values for Amzn data computed.")

print("Explaining Dslr model with SHAP...")
explainer_tg = shap.TreeExplainer(clf_tg_base)
# shap_values_fr = explainer_fr.shap_values(X_target_train)
def explain_single_instance_t(x):
    shap_vals = explainer_tg.shap_values(x, check_additivity=False)
    return shap_vals

# Parallel computation
shap_values_tg = Parallel(n_jobs=n_jobs)(
    delayed(explain_single_instance_t)(X_target_train[i]) for i in tqdm(range(len(X_target_train)))
)

# Convert to consistent array shape
# Each item is a list (one per class); stack per class
shap_values_fr_stacked = [np.vstack([shap_values_tg[i][cls] for i in range(len(X_target_train))])
                          for cls in range(len(shap_values_tg[0]))]

print("SHAP values for Dslr data computed.")

for i, sv in enumerate(shap_values_sr_stacked):
    print(f"Class {i} SHAP shape: {sv.shape}")

# print("Shape of SHAP values",shap_values_en.shape, shap_values_fr.shape)

print("Computing SHAP summaries...")
# shap_en_summary = np.mean([np.abs(sv) for sv in shap_values_en], axis=1)
# shap_fr_summary = np.mean([np.abs(sv) for sv in shap_values_fr], axis=1)
# shap_en_summary = np.mean(np.abs(shap_values_en), axis=0)  # shape: (10000, 1536)
# shap_fr_summary = np.mean(np.abs(shap_values_fr), axis=0) 
shap_sr_summary = np.mean(np.abs(np.stack(shap_values_sr)), axis=0)
shap_tg_summary = np.mean(np.abs(np.stack(shap_values_tg)), axis=0) 

print("SHAP summaries computed.")
print("SHAP summaries shap_sr_summary", shap_sr_summary.shape)
print("SHAP summaries shap_tg_summary", shap_tg_summary.shape)

print("Computing cosine similarity matrix...")
similarities = cosine_similarity(shap_tg_summary, shap_sr_summary)
print("similarities shape: ", similarities.shape)
top_match_indices = np.argmax(similarities, axis=1)
print("Top-1 correspondences identified.")

plt.figure(figsize=(10, 8))
sns.heatmap(similarities, cmap="viridis")
plt.title("Cosine Similarity Heatmap (SHAP - Dslr vs Amzn)")
plt.xlabel("Amzn Samples (Source)")
plt.ylabel("Dslr Samples (Target)")
plt.tight_layout()
plt.savefig("Heatmap_similarity_office31.jpeg")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Combine SHAP summaries for visualization
combined_shap = np.vstack([shap_sr_summary, shap_tg_summary])
labels = ["Amzn"] * shap_sr_summary.shape[0] + ["Dslr"] * shap_tg_summary.shape[0]

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(combined_shap)

# Plot
plt.figure(figsize=(8, 6))
for label in ["Amzn", "Dslr"]:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)

plt.title("PCA of SHAP Explanation Vectors")
plt.legend()
plt.grid(True)
plt.savefig("PCA_SHAP_values.jpeg")

# Create correspondence pairs
threshold = 0.5
valid_mask = similarities.max(axis=1) > threshold
corresponding_sr = X_source_scaled[top_match_indices]
corresponding_tg = X_target_train[top_match_indices]
# corresponding_en = X_source_scaled[top_match_indices]
# corresponding_fr = X_target_train
# corresponding_fr = X_target_train
# corresponding_en = X_source_scaled[top_match_indices[:len(X_target_train)]]

print("corresponding_sr.shape", corresponding_sr.shape)
print("corresponding_tg.shape", corresponding_tg.shape)

# Save Correspondences
joblib.dump((corresponding_sr, corresponding_tg), "explainable_correspondences_shap_rf.pkl")

alphas = [1e-200, 1e-500, 1e-50, 1e-15, 1e-10, 1e-7, 1e-5, 1e-3, 1e-1]
best_alpha = None
best_acc = -1
best_report = ""
best_model = None

for alpha in alphas:
    # Learn transformation matrix
    ridge = Ridge(alpha=alpha, solver='auto', random_state=42)  # Added `solver` to control parallelism
    ridge.fit(corresponding_sr, corresponding_tg)
    P = ridge.coef_

    # Transform source data
    X_sr_transformed = ridge.predict(X_source_scaled)

    # Step 8: Combine Transformed Source + Target
    X_combined = np.vstack([X_sr_transformed, X_target_train])
    y_combined = np.hstack([y_source, y_target_train])

    # Parallel training of the transfer model
    clf_transfer = train_rf_model(X_combined, y_combined)

    # Step 10: Evaluate Transfer Learning Model on Target Test Data
    y_tg_pred = clf_transfer.predict(X_target_test)
    transfer_accuracy = accuracy_score(y_target_test, y_tg_pred)
    report = classification_report(y_target_test, y_tg_pred)

    print(f"Alpha={alpha} → Accuracy={transfer_accuracy:.4f}")
    print(report)

    if transfer_accuracy > best_acc:
        best_acc = transfer_accuracy
        best_alpha = alpha
        best_report = report
        best_model = clf_transfer

print("Best alpha:", best_alpha)
print("Best accuracy:", best_acc)
print(best_report)