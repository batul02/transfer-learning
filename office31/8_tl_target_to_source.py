import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

data = torch.load("extracted_features.pt")
X_amazon = data["X_amazon_feats"]
y_amazon = data["y_amazon"]
X_dslr = data["X_dslr_feats"]
y_dslr = data["y_dslr"]

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

# Evaluate on Transformed Target
print("\nEvaluating baseline source model on transformed target test set...")
y_pred_target = baseline_model.predict(X_target_transformed)

print("Accuracy after Transformation:", accuracy_score(y_dslr, y_pred_target))
print("Classification Report:")
print(classification_report(y_dslr, y_pred_target, zero_division=0))