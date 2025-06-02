import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from collections import Counter

data = np.load("extracted_features_label_enc_feat_scl.npz")

# Extract arrays
X_amazon = data["X_src"]   # Source feature matrix
y_amazon = data["y_src"]   # Source labels

X_dslr = data["X_tgt"]   # Target feature matrix
y_dslr = data["y_tgt"]
y_amazon_np = y_amazon.numpy() if isinstance(y_amazon, torch.Tensor) else np.array(y_amazon)
y_dslr_np = y_dslr.numpy() if isinstance(y_dslr, torch.Tensor) else np.array(y_dslr)
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

data = torch.load("office31_feature_correspondence.pt", weights_only=False)

# Extract the matrices
source_matrix = data["source_matrix"]
target_matrix = data["target_matrix"]

baseline_model = joblib.load("best_source_model_dslr.pkl")

print("\nFitting Ridge Regression (Target → Source)...")
ridge = Ridge(alpha=1e-5)
ridge.fit(target_matrix, source_matrix)
P_matrix = ridge.coef_

# Transform Target Test
X_target_transformed =  np.matmul(np.asarray(X_dslr), np.asarray(P_matrix.T)) 
le = joblib.load("label_encoder.pkl")

# Evaluate on Transformed Target
print("\nEvaluating baseline source model on transformed target test set...")
y_pred_target = baseline_model.predict(X_target_transformed)
y_target_test = le.inverse_transform(y_pred_target)
y_dslr = le.inverse_transform(y_dslr)
print("Accuracy after Transformation:", accuracy_score(y_dslr, y_pred_target))
print("Classification Report:")
print(classification_report(y_dslr, y_pred_target, zero_division=0))