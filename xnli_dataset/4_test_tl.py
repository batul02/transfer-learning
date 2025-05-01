import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

data = np.load("encoded_data.npz")
X_en = data["X_en"]
y_en = data["y_en"]

X_source_scaled = np.load("X_source_scaled.npy")
# X_target_scaled = np.load("X_target_scaled.npy")

print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
# print("Shape of Target Dataset scalled: ", X_target_scaled.shape)

data = np.load("target_train_test_split.npz")
X_target_train = data["X_target_train"]
X_target_test = data["X_target_test"]
y_target_train = data["y_target_train"]
y_target_test = data["y_target_test"]

P_matrix = np.load("P_matrix.npy")
print("Shape of P_matrix: ", P_matrix.shape)

random_search = joblib.load("random_search_rf.pkl")

X_source_transformed = X_source_scaled @ P_matrix.T

X_merged = np.vstack([X_source_transformed, X_target_train])
y_merged = np.hstack([y_en, y_target_train])

best_params = random_search.best_params_
final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_merged, y_merged)

y_pred = final_model.predict(X_target_test)

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_target_test, y_pred))
print(classification_report(y_target_test, y_pred))

