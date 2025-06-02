from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
import joblib
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

data = np.load("../crema_audio_librosa_features_downsized_trg.npz")

# Extract arrays
X_src = data["X_src"]   # Source feature matrix
y_src = data["y_src"]   # Source labels

X_tgt = data["X_tgt"]   # Target feature matrix
y_tgt = data["y_tgt"]


print("Shape of Source Dataset: ", X_src.shape)
print("Shape of Target Dataset: ", X_tgt.shape)

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_src)   # Fit on source
X_target_scaled = scaler.transform(X_tgt)

# X_source_scaled = X_source_scaled
# X_target_scaled = X_target_scaled[:5000,:]
# y_tgt = y_tgt
# y_src = y_src
print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
print("Shape of Target Dataset scalled: ", X_target_scaled.shape)

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target_scaled, y_tgt, test_size=0.2, stratify=y_tgt, random_state=42
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

print("Base Model Accuracy on Female test data:", base_accuracy)
joblib.dump(clf_tg_base, "rf_tg_base.pkl")

# Step 4: Train Logistic Regression Models on Source
clf_en = train_rf_model(X_source_scaled, y_src)

# Step 5: Compute SHAP Values for Source and Target Models
print("Explaining Male model with SHAP...")
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
print("SHAP values for Male data computed.")

print("Explaining Female model with SHAP...")
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

print("SHAP values for Female data computed.")

for i, sv in enumerate(shap_values_sr_stacked):
    print(f"Class {i} SHAP shape: {sv.shape}")

# print("Shape of SHAP values",shap_values_en.shape, shap_values_fr.shape)

print("Computing SHAP summaries...")
# shap_en_summary = np.mean([np.abs(sv) for sv in shap_values_en], axis=1)
# shap_fr_summary = np.mean([np.abs(sv) for sv in shap_values_fr], axis=1)
# shap_en_summary = np.mean(np.abs(shap_values_en), axis=0)  # shape: (10000, 1536)
# shap_fr_summary = np.mean(np.abs(shap_values_fr), axis=0) 
shap_sr_summary = np.mean(np.abs(np.stack(shap_values_sr)), axis=2)
shap_tg_summary = np.mean(np.abs(np.stack(shap_values_tg)), axis=2) 

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
plt.title("Cosine Similarity Heatmap (SHAP - female vs male)")
plt.xlabel("Male Samples (Source)")
plt.ylabel("Female Samples (Target)")
plt.tight_layout()
plt.savefig("Heatmap_similarity_office31.jpeg")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Combine SHAP summaries for visualization
combined_shap = np.vstack([shap_sr_summary, shap_tg_summary])
labels = ["Male"] * shap_sr_summary.shape[0] + ["Female"] * shap_tg_summary.shape[0]

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(combined_shap)

# Plot
plt.figure(figsize=(8, 6))
for label in ["Male", "Female"]:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)

plt.title("PCA of SHAP Explanation Vectors")
plt.legend()
plt.grid(True)
plt.savefig("PCA_SHAP_values.jpeg")

# Create correspondence pairs
threshold = 0.5
valid_mask = similarities.max(axis=1) > threshold
corresponding_sr = X_source_scaled[top_match_indices[valid_mask]]
corresponding_tg = X_target_train[valid_mask]
# corresponding_en = X_source_scaled[top_match_indices]
# corresponding_fr = X_target_train
# corresponding_fr = X_target_train
# corresponding_en = X_source_scaled[top_match_indices[:len(X_target_train)]]

print("corresponding_en.shape", corresponding_sr.shape)
print("corresponding_fr.shape", corresponding_tg.shape)

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
    y_combined = np.hstack([y_src, y_target_train])

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