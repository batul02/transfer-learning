from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

# Load data
# data = np.load("../encoded_data.npz")
# X_en, y_en = data["X_en"], data["y_en"]
# X_fr, y_fr = data["X_fr"], data["y_fr"]
data = np.load("../encoded_data.npz")
X_en, y_en = data["X_en"][:10000,:], data["y_en"][:10000]
X_fr, y_fr = data["X_fr"][:5000], data["y_fr"][:5000]

# Scale data
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_en)
X_target_scaled = scaler.transform(X_fr)

# Split target
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target_scaled, y_fr, test_size=0.2, stratify=y_fr, random_state=42
)

# Base RF model trained only on target train
clf_fr_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_fr_base.fit(X_target_train, y_target_train)
print("Base Accuracy:", accuracy_score(y_target_test, clf_fr_base.predict(X_target_test)))
joblib.dump(clf_fr_base, "clf_fr_base_rf_lime.pkl")

# RF models on source and target
rf_en = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_en.fit(X_source_scaled, y_en)

rf_fr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fr.fit(X_target_train, y_target_train)

# Lime explainers
explainer_en = LimeTabularExplainer(X_source_scaled, feature_names=[f'f{i}' for i in range(X_source_scaled.shape[1])],
                                     class_names=np.unique(y_en), verbose=False, mode='classification')

explainer_fr = LimeTabularExplainer(X_target_train, feature_names=[f'f{i}' for i in range(X_target_scaled.shape[1])],
                                     class_names=np.unique(y_fr), verbose=False, mode='classification')

# LIME explanation vectors
def compute_lime_summary(explainer, model, X, num_features=1536, n_jobs=-1):
    """
    Parallelized LIME explanation computation.
    """

    def explain_one(i):
        exp = explainer.explain_instance(X[i], model.predict_proba, num_features=num_features)
        weights = np.zeros(X.shape[1])
        for idx, w in exp.as_list():
            try:
                idx_parts = idx.split()
                if len(idx_parts) > 3:
                    feat_index = int(idx_parts[2][1:])
                else:
                    feat_index = int(idx_parts[0][1:])
                weights[feat_index] = abs(w)
            except ValueError:
                return np.zeros(X.shape[1])  # Skip if invalid
        return weights

    # Parallel over all samples
    results = Parallel(n_jobs=32, verbose=10)(
        delayed(explain_one)(i) for i in tqdm(range(len(X)))
    )

    return np.vstack(results)

# Subset for speed
print("Computing LIME for English...")
lime_en_summary = compute_lime_summary(explainer_en, rf_en, X_source_scaled)
print("Computing LIME for French...")
lime_fr_summary = compute_lime_summary(explainer_fr, rf_fr, X_target_train)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Combine for visualization
combined_lime = np.vstack([lime_en_summary, lime_fr_summary])
labels = ["EN"] * lime_en_summary.shape[0] + ["FR"] * lime_fr_summary.shape[0]

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(combined_lime)

# Plot
plt.figure(figsize=(8, 6))
for label in ["EN", "FR"]:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)

plt.title("PCA of LIME Explanation Vectors")
plt.legend()
plt.grid(True)
plt.savefig("PCA_LIME_values.jpeg")

# Cosine similarity to find correspondences
similarities = cosine_similarity(lime_fr_summary, lime_en_summary)
top_match_indices = np.argmax(similarities, axis=1)
threshold = 0.5
valid_mask = similarities.max(axis=1) > threshold
corresponding_en = X_source_scaled[top_match_indices[valid_mask]]
corresponding_fr = X_target_train[valid_mask]

joblib.dump((corresponding_en, corresponding_fr), "lime_correspondences_rf.pkl")

# Learn transformation
ridge = Ridge(alpha=1e-3)
ridge.fit(corresponding_en, corresponding_fr)
X_en_transformed = ridge.predict(X_source_scaled)

# Merge transformed source + target train
X_combined = np.vstack([X_en_transformed, X_target_train])
y_combined = np.hstack([y_en, y_target_train])

# Final RF transfer model
clf_transfer = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_transfer.fit(X_combined, y_combined)

# Evaluate on target test
transfer_accuracy = accuracy_score(y_target_test, clf_transfer.predict(X_target_test))
print("Transfer Learning Accuracy:", transfer_accuracy)
