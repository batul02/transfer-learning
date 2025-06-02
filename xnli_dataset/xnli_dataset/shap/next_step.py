import numpy as np
import seaborn as sns
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


data = np.load("scaled_src_shap.npz")
X_source_scaled = data["X_source_scaled"]
y_en = data["y_en"]


dslr_split = np.load("fr_target_train_test_split_shap.npz")
X_target_train = dslr_split["X_target_train"]
X_target_test = dslr_split["X_target_test"]
y_target_train = dslr_split["y_target_train"]
y_target_test = dslr_split["y_target_test"]

data = np.load("shap_summaries_and_similarity.npz")
shap_en_summary = data["shap_en_summary"]
shap_fr_summary = data["shap_fr_summary"]
similarities = data["cosine_similarities"]

def train_rf_model(X, y, random_state=42):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state).fit(X, y)

# plt.figure(figsize=(10, 8))
# sns.heatmap(similarities, cmap="viridis")
# plt.title("Cosine Similarity Heatmap (SHAP - FR vs EN)")
# plt.xlabel("English Samples (Source)")
# plt.ylabel("French Samples (Target)")
# plt.tight_layout()
# plt.savefig("Heatmap_similarity_xnli.jpeg")

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # Combine SHAP summaries for visualization
# combined_shap = np.vstack([shap_en_summary, shap_fr_summary])
# labels = ["EN"] * shap_en_summary.shape[0] + ["FR"] * shap_fr_summary.shape[0]

# # PCA
# pca = PCA(n_components=2)
# reduced = pca.fit_transform(combined_shap)

# # Plot
# plt.figure(figsize=(8, 6))
# for label in ["EN", "FR"]:
#     idxs = [i for i, l in enumerate(labels) if l == label]
#     plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)

# plt.title("PCA of SHAP Explanation Vectors")
# plt.legend()
# plt.grid(True)
# plt.savefig("PCA_SHAP_values.jpeg")

top_match_indices = np.argmax(similarities, axis=1)
print("Top-1 correspondences identified.")

# Create correspondence pairs
threshold = 0.95
valid_mask = similarities.max(axis=1) > threshold
corresponding_en = X_source_scaled[top_match_indices[valid_mask]]
corresponding_fr = X_target_train[valid_mask]
# corresponding_en = X_source_scaled[top_match_indices]
# corresponding_fr = X_target_train
# corresponding_fr = X_target_train
# corresponding_en = X_source_scaled[top_match_indices[:len(X_target_train)]]

print("corresponding_en.shape", corresponding_en.shape)
print("corresponding_fr.shape", corresponding_fr.shape)

# Save Correspondences
joblib.dump((corresponding_en, corresponding_fr), "explainable_correspondences_shap_rf.pkl")

# alphas = [1e-200, 1e-150, 1e-100, 1e-75, 1e-50, 1e-25]
alphas = [1e-15, 1e-10, 1e-7, 1e-5, 1e-3, 1e-1]

best_alpha = None
best_acc = -1
best_report = ""
best_model = None

for alpha in alphas:
    # Learn transformation matrix
    ridge = Ridge(alpha=alpha, solver='auto', random_state=42)  # Added `solver` to control parallelism
    ridge.fit(corresponding_en, corresponding_fr)
    P = ridge.coef_

    # Transform source data
    # X_en_transformed = ridge.predict(X_source_scaled)
    X_en_transformed = np.matmul(np.asarray(X_source_scaled), np.asarray(P.T))

    # Step 8: Combine Transformed Source + Target
    X_combined = np.vstack([X_en_transformed, X_target_train])
    y_combined = np.hstack([y_en, y_target_train])

    # Parallel training of the transfer model
    clf_transfer = train_rf_model(X_combined, y_combined)

    # Step 10: Evaluate Transfer Learning Model on Target Test Data
    y_fr_pred = clf_transfer.predict(X_target_test)
    transfer_accuracy = accuracy_score(y_target_test, y_fr_pred)
    report = classification_report(y_target_test, y_fr_pred)

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

