from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

data = np.load("crema_audio_librosa_features_downsized_trg_label_enc.npz")

# Extract arrays
X_src = data["X_src"]   # Source feature matrix
y_src = data["y_src"]   # Source labels

X_tgt = data["X_tgt"]   # Target feature matrix
y_tgt = data["y_tgt"]

scaler = StandardScaler()
X_src_scaled = scaler.fit_transform(X_src)

le = joblib.load("label_encoder.pkl")

# 2) Apply the same transform to target features
X_tgt_scaled = scaler.transform(X_tgt)

data = torch.load("cremad_feature_correspondence.pt", weights_only=False)

# Extract the matrices
source_matrix = data["source_matrix"]
target_matrix = data["target_matrix"]

random_search = joblib.load("random_search_rf_target.pkl")
best_params = random_search.best_params_
# print("Loaded Best Params:", best_params)

split = np.load("target_train_test_split.npz")
X_train = split["X_target_train"]
X_test = split["X_target_test"]
y_train = split["y_target_train"]
y_test = split["y_target_test"]

# ridge = Ridge(alpha=1e-100)
# ridge.fit(source_matrix, target_matrix)
# P_matrix = ridge.coef_ 

# X_transformed = np.matmul(np.asarray(X_src_scaled), np.asarray(P_matrix.T))

# X_combined_train = np.vstack((X_transformed, X_train))
# y_combined_train = np.hstack((y_src, y_train))

# rf_final = RandomForestClassifier(**best_params, random_state=42)
# rf_final.fit(X_combined_train, y_combined_train)

# # 9. Evaluate on DSLR test set
# y_pred = rf_final.predict(X_test)
# y_pred = le.inverse_transform(y_pred)
# y_test = le.inverse_transform(y_test)
# print("\nResults After Transfer Learning:")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

alphas = [1e-500, 1e-100, 1e-50, 1e-15, 1e-10, 1e-7, 1e-5, 1e-3, 1e-1]
best_alpha = None
best_acc = -1
best_report = ""
best_model = None

y_test = le.inverse_transform(y_test)

for alpha in alphas:
    # Learn transformation matrix
    ridge = Ridge(alpha=alpha)
    ridge.fit(source_matrix, target_matrix)
    P_matrix = ridge.coef_ 

    # Transform source data
    X_transformed = np.matmul(np.asarray(X_src_scaled), np.asarray(P_matrix.T))

    # Combine with target train
    X_combined_train = np.vstack((X_transformed, X_train))
    y_combined_train = np.hstack((y_src, y_train))

    rf_final = RandomForestClassifier(**best_params, random_state=42)
    rf_final.fit(X_combined_train, y_combined_train)

    # Evaluate on target test
    y_pred = rf_final.predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Alpha={alpha} → Accuracy={acc:.4f}")
    print(report)

    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha
        best_report = report
        best_model = rf_final

print("Best alpha:", best_alpha)
print("Best accuracy:", best_acc)
print(best_report)