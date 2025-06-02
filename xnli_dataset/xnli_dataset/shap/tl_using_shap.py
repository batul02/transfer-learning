from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
import joblib
from sklearn.metrics import accuracy_score

data = np.load("../encoded_data.npz")
X_en = data["X_en"]
y_en = data["y_en"]
X_fr = data["X_fr"]
y_fr = data["y_fr"]


print("Shape of Source Dataset: ", X_en.shape)
print("Shape of Target Dataset: ", X_fr.shape)

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_en)   # Fit on source
X_target_scaled = scaler.transform(X_fr) 

print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
print("Shape of Target Dataset scalled: ", X_target_scaled.shape)

# Step 2: Train-Test Split for Target Data
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target_scaled, y_fr, test_size=0.2, stratify=y_fr, random_state=42
)

# Parallelize the training process using joblib
n_jobs = -1  # Use the maximum number of available cores

# Step 3: Train Base Model on Target Train Data
def train_base_model(X_train, y_train, random_state=42):
    clf_fr_base = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=n_jobs)
    clf_fr_base.fit(X_train, y_train)
    return clf_fr_base

# Parallel training of the base model (only one as per your case)
clf_fr_base = train_base_model(X_target_train, y_target_train)

# Step 11: Evaluate Base Model on Target Test Data
y_fr_base_pred = clf_fr_base.predict(X_target_test)
base_accuracy = accuracy_score(y_target_test, y_fr_base_pred)

print("Base Model Accuracy on French test data:", base_accuracy)
joblib.dump(clf_fr_base, "clf_fr_base.pkl")

# Step 4: Train Logistic Regression Models on Source
clf_en = LogisticRegression(max_iter=1000, n_jobs=n_jobs)
clf_en.fit(X_source_scaled, y_en)

# Step 5: Compute SHAP Values for Source and Target Models
explainer_en = shap.Explainer(clf_en, X_source_scaled)
shap_values_en = explainer_en(X_source_scaled)

explainer_fr = shap.Explainer(clf_fr_base, X_target_train)
shap_values_fr = explainer_fr(X_target_train)

print("Shape of SHAP values",shap_values_en.shape, shap_values_fr.shape)

# Step 6: Find Top Correspondences Using SHAP
# shap_en_summary = np.abs(shap_values_en.values).mean(axis=0)
# shap_fr_summary = np.abs(shap_values_fr.values).mean(axis=0)

shap_en_summary = np.abs(shap_values_en.values).mean(axis=2)  # → shape (1536,)
shap_fr_summary = np.abs(shap_values_fr.values).mean(axis=2)

print("SHAP summaries shap_en_summary", shap_en_summary.shape)
print("SHAP summaries shap_fr_summary", shap_fr_summary.shape)

# Find top-1 correspondence for each French example
similarities = cosine_similarity(shap_fr_summary, shap_en_summary)
top_match_indices = np.argmax(similarities, axis=1)

# Create correspondence pairs
threshold = 0.7
valid_mask = similarities.max(axis=1) > threshold
corresponding_en = X_source_scaled[top_match_indices[valid_mask]]
corresponding_fr = X_target_train[valid_mask]
# corresponding_en = X_source_scaled[top_match_indices]
# corresponding_fr = X_target_train

# Save Correspondences
joblib.dump((corresponding_en, corresponding_fr), "explainable_correspondences_shap.pkl")

# Step 7: Learn Transformation Matrix with Ridge
ridge = Ridge(alpha=1.0, solver='auto')  # Added `solver` to control parallelism
ridge.fit(corresponding_en, corresponding_fr)

# Transform Source Data
X_en_transformed = ridge.predict(X_source_scaled)

# Step 8: Combine Transformed Source + Target
X_combined = np.vstack([X_en_transformed, X_target_train])
y_combined = np.hstack([y_en, y_target_train])

# Step 9: Train Transfer Model on Combined Data
def train_transfer_model(X_combined, y_combined):
    clf_transfer = LogisticRegression(max_iter=1000, n_jobs=n_jobs)
    clf_transfer.fit(X_combined, y_combined)
    return clf_transfer

# Parallel training of the transfer model
clf_transfer = train_transfer_model(X_combined, y_combined)

# Step 10: Evaluate Transfer Learning Model on Target Test Data
y_fr_pred = clf_transfer.predict(X_target_test)
transfer_accuracy = accuracy_score(y_target_test, y_fr_pred)

print("Transfer Learning Accuracy on French test data:", transfer_accuracy)