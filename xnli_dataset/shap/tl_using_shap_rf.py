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
X_en = data["X_en"][:10000,:]
y_en = data["y_en"][:10000]
X_fr = data["X_fr"][:5000,:]
y_fr = data["y_fr"][:5000]


print("Shape of Source Dataset: ", X_en.shape)
print("Shape of Target Dataset: ", X_fr.shape)

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_en)   # Fit on source
X_target_scaled = scaler.transform(X_fr)

# X_source_scaled = X_source_scaled
# X_target_scaled = X_target_scaled[:5000,:]
# y_fr = y_fr
# y_en = y_en
print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
print("Shape of Target Dataset scalled: ", X_target_scaled.shape)

# Step 2: Train-Test Split for Target Data
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target_scaled, y_fr, test_size=0.2, stratify=y_fr, random_state=42
)

# Parallelize the training process using joblib
n_jobs = -1  # Use the maximum number of available cores

# Step 3: Train Base Model on Target Train Data
from sklearn.ensemble import RandomForestClassifier

def train_rf_model(X, y, random_state=42):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state).fit(X, y)

# Parallel training of the base model (only one as per your case)
clf_fr_base = train_rf_model(X_target_train, y_target_train)

# Step 11: Evaluate Base Model on Target Test Data
y_fr_base_pred = clf_fr_base.predict(X_target_test)
base_accuracy = accuracy_score(y_target_test, y_fr_base_pred)

print("Base Model Accuracy on French test data:", base_accuracy)
joblib.dump(clf_fr_base, "rf_fr_base.pkl")

# Step 4: Train Logistic Regression Models on Source
clf_en = train_rf_model(X_source_scaled, y_en)

# Step 5: Compute SHAP Values for Source and Target Models
print("Explaining English model with SHAP...")
explainer_en = shap.TreeExplainer(clf_en)
shap_values_en = explainer_en.shap_values(X_source_scaled)
print("SHAP values for English data computed.")

print("Explaining French model with SHAP...")
explainer_fr = shap.TreeExplainer(clf_fr_base)
shap_values_fr = explainer_fr.shap_values(X_target_train)
print("SHAP values for French data computed.")

print("Shape of SHAP values",shap_values_en.shape, shap_values_fr.shape)

print("Computing SHAP summaries...")
# shap_en_summary = np.mean([np.abs(sv) for sv in shap_values_en], axis=1)
# shap_fr_summary = np.mean([np.abs(sv) for sv in shap_values_fr], axis=1)
# shap_en_summary = np.mean(np.abs(shap_values_en), axis=0)  # shape: (10000, 1536)
# shap_fr_summary = np.mean(np.abs(shap_values_fr), axis=0) 
shap_en_summary = np.mean(np.abs(np.stack(shap_values_en)), axis=2)
shap_fr_summary = np.mean(np.abs(np.stack(shap_values_fr)), axis=2) 

print("SHAP summaries computed.")
print("SHAP summaries shap_en_summary", shap_en_summary.shape)
print("SHAP summaries shap_fr_summary", shap_fr_summary.shape)

print("Computing cosine similarity matrix...")
similarities = cosine_similarity(shap_fr_summary, shap_en_summary)
print("similarities shape: ", similarities.shape)
top_match_indices = np.argmax(similarities, axis=1)
print("Top-1 correspondences identified.")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Combine SHAP summaries for visualization
combined_shap = np.vstack([shap_en_summary, shap_fr_summary])
labels = ["EN"] * shap_en_summary.shape[0] + ["FR"] * shap_fr_summary.shape[0]

# PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(combined_shap)

# Plot
plt.figure(figsize=(8, 6))
for label in ["EN", "FR"]:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, alpha=0.6)

plt.title("PCA of SHAP Explanation Vectors")
plt.legend()
plt.grid(True)
plt.savefig("PCA_SHAP_values.jpeg")

# Create correspondence pairs
threshold = 0.5
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

# Step 7: Learn Transformation Matrix with Ridge
ridge = Ridge(alpha=1e-5, solver='auto', random_state=42)  # Added `solver` to control parallelism
ridge.fit(corresponding_en, corresponding_fr)

# Transform Source Data
X_en_transformed = ridge.predict(X_source_scaled)

# Step 8: Combine Transformed Source + Target
X_combined = np.vstack([X_en_transformed, X_target_train])
y_combined = np.hstack([y_en, y_target_train])

# Parallel training of the transfer model
clf_transfer = train_rf_model(X_combined, y_combined)

# Step 10: Evaluate Transfer Learning Model on Target Test Data
y_fr_pred = clf_transfer.predict(X_target_test)
transfer_accuracy = accuracy_score(y_target_test, y_fr_pred)

print("Transfer Learning Accuracy on French test data:", transfer_accuracy)