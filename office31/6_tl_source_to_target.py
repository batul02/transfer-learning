import numpy as np
import torch
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

data = torch.load("extracted_features.pt")
X_amazon = data["X_amazon_feats"]
y_amazon = data["y_amazon"]
X_dslr = data["X_dslr_feats"]
y_dslr = data["y_dslr"]

data = torch.load("office31_feature_correspondence.pt", weights_only=False)

# Extract the matrices
source_matrix = data["source_matrix"]
target_matrix = data["target_matrix"]

random_search = joblib.load("random_search_rf_dslr.pkl")
best_params = random_search.best_params_
# print("Loaded Best Params:", best_params)

dslr_split = np.load("dslr_target_train_test_split.npz")
X_dslr_train = dslr_split["X_target_train"]
X_dslr_test = dslr_split["X_target_test"]
y_dslr_train = dslr_split["y_target_train"]
y_dslr_test = dslr_split["y_target_test"]

ridge = Ridge(alpha=1e-10)
ridge.fit(source_matrix, target_matrix)
P_matrix = ridge.coef_ 

X_amazon_transformed = np.matmul(np.asarray(X_amazon), np.asarray(P_matrix.T)) 

X_combined_train = np.vstack((X_amazon_transformed, X_dslr_train))
y_combined_train = np.hstack((y_amazon, y_dslr_train))

rf_final = RandomForestClassifier(**best_params, random_state=42)
rf_final.fit(X_combined_train, y_combined_train)

# 9. Evaluate on DSLR test set
y_pred = rf_final.predict(X_dslr_test)

print("\nResults After Transfer Learning:")
print("Accuracy:", accuracy_score(y_dslr_test, y_pred))
print("Classification Report:\n", classification_report(y_dslr_test, y_pred))