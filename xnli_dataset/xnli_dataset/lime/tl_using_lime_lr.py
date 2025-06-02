from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
data = np.load("../encoded_data.npz")
X_en, y_en = data["X_en"][:10000,:], data["y_en"][:10000]
X_fr, y_fr = data["X_fr"][:5000], data["y_fr"][:5000]

print("Shape of Source Dataset: ", X_en.shape)
print("Shape of Target Dataset: ", X_fr.shape)

# Scaling
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_en)
X_target_scaled = scaler.transform(X_fr)

print("Shape of Source Dataset scaled: ", X_source_scaled.shape)
print("Shape of Target Dataset scaled: ", X_target_scaled.shape)

# Target split
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target_scaled, y_fr, test_size=0.2, stratify=y_fr, random_state=42
)

# Base model
clf_fr_base = LogisticRegression(max_iter=3000, random_state=42, n_jobs=-1)
clf_fr_base.fit(X_target_train, y_target_train)
print("Base Accuracy:", accuracy_score(y_target_test, clf_fr_base.predict(X_target_test)))
joblib.dump(clf_fr_base, "clf_fr_base_lime.pkl")

# Source model
# clf_en = LogisticRegression(max_iter=1000, n_jobs=-1)
# clf_en.fit(X_source_scaled, y_en)

# # LIME Explainers
# explainer_en = LimeTabularExplainer(X_source_scaled, feature_names=[f'f{i}' for i in range(X_source_scaled.shape[1])],
#                                      class_names=np.unique(y_en), verbose=False, mode='classification')
# explainer_fr = LimeTabularExplainer(X_target_train, feature_names=[f'f{i}' for i in range(X_target_scaled.shape[1])],
#                                      class_names=np.unique(y_fr), verbose=False, mode='classification')

# Compute LIME explanations (mean absolute coefficients)
# def compute_lime_summary(explainer, model, X, num_features=1536):
#     weights_matrix = []
#     for i in range(len(X)):
#         exp = explainer.explain_instance(X[i], model.predict_proba, num_features=num_features)
#         weights = np.zeros(X.shape[1])
#         for idx, w in exp.as_list():
#             try:
#                 # Extract the numeric feature index from idx, even if it has conditions like '<= -0.66'
#                 # Example: '0.01 < f832 <= 0.68' -> Extract '832'
#                 idx_parts = idx.split()
#                 if len(idx_parts) > 3:
#                     # The second part will be the feature index, e.g., 'f832'
#                     feat_index = int(idx_parts[2][1:])  # Remove the 'f' and convert to int
#                 else:
#                     feat_index = int(idx_parts[0][1:])  # In case it's just 'f832'
                
#                 # Store the weight in the correct feature index
#                 weights[feat_index] = abs(w)
#             except ValueError:
#                 print(f"Skipping invalid idx format: {idx}")  # Debugging: print invalid feature name
#                 continue  # Skip invalid feature names

#         weights_matrix.append(weights)
#     return np.array(weights_matrix)

# def compute_lime_summary(explainer, model, X, num_features=1536, n_jobs=-1):
#     """
#     Parallelized LIME explanation computation.
#     """

#     def explain_one(i):
#         exp = explainer.explain_instance(X[i], model.predict_proba, num_features=num_features)
#         weights = np.zeros(X.shape[1])
#         for idx, w in exp.as_list():
#             try:
#                 idx_parts = idx.split()
#                 if len(idx_parts) > 3:
#                     feat_index = int(idx_parts[2][1:])
#                 else:
#                     feat_index = int(idx_parts[0][1:])
#                 weights[feat_index] = abs(w)
#             except ValueError:
#                 return np.zeros(X.shape[1])  # Skip if invalid
#         return weights

#     # Parallel over all samples
#     results = Parallel(n_jobs=n_jobs, verbose=10)(
#         delayed(explain_one)(i) for i in tqdm(range(len(X)))
#     )

#     return np.vstack(results)

# # Run LIME (this can take time)
# print("Computing LIME explanations for English data...")
# lime_en_summary = compute_lime_summary(explainer_en, clf_en, X_source_scaled)  # Use subset if slow
# print("Computing LIME explanations for French data...")
# lime_fr_summary = compute_lime_summary(explainer_fr, clf_fr_base, X_target_train)

# # Cosine Similarity & Correspondences
# similarities = cosine_similarity(lime_fr_summary, lime_en_summary)
# top_match_indices = np.argmax(similarities, axis=1)
# threshold = 0.5
# valid_mask = similarities.max(axis=1) > threshold
# corresponding_en = X_source_scaled[top_match_indices[valid_mask]]
# corresponding_fr = X_target_train[valid_mask]

# joblib.dump((corresponding_en, corresponding_fr), "lime_correspondences.pkl")

corresponding_en, corresponding_fr = joblib.load("lime_correspondences.pkl")

print(np.all(corresponding_en == 0))
print(np.all(corresponding_fr == 0))

print("Shape of loaded English correspondences:", corresponding_en.shape)
print("Shape of loaded French correspondences:", corresponding_fr.shape)

# Learn transformation
ridge = Ridge(alpha=1e-1)
ridge.fit(corresponding_en, corresponding_fr)
X_en_transformed = ridge.predict(X_source_scaled)

# Transfer model
X_combined = np.vstack([X_en_transformed, X_target_train])
y_combined = np.hstack([y_en, y_target_train])
clf_transfer = LogisticRegression(max_iter=3000, n_jobs=-1)
clf_transfer.fit(X_combined, y_combined)

# Evaluate
transfer_accuracy = accuracy_score(y_target_test, clf_transfer.predict(X_target_test))
print("Transfer Learning Accuracy:", transfer_accuracy)