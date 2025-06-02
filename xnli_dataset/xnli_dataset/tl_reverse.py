import numpy as np
import joblib

from sklearn.metrics import accuracy_score, classification_report

data = np.load("encoded_data.npz")
X_en = data["X_en"]
y_en = data["y_en"]
X_fr = data["X_fr"]
y_fr = data["y_fr"]

X_source_scaled = np.load("X_source_scaled.npy")
X_target_scaled = np.load("X_target_scaled.npy")

print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
print("Shape of Target Dataset scalled: ", X_target_scaled.shape)

P_matrix = np.load("P_matrix.npy")
print("Shape of P_matrix: ", P_matrix.shape)

source_matrix = np.load("source_matrix.npy")
target_matrix = np.load("target_matrix.npy")

#transforming target to source
from sklearn.linear_model import Ridge

# Learn a new transformation: Target → Source
ridge_reverse = Ridge(alpha=1e-5)
ridge_reverse.fit(target_matrix, source_matrix)

P_reverse = ridge_reverse.coef_  # Transformation matrix (target to source)
bias_reverse = ridge_reverse.intercept_

if not(np.all(P_matrix == 0)):
    # np.save("P_matrix_reversed_1e-8.npy", P_reverse)
    print("saved")

# Apply transformation to target data
transformed_target = X_target_scaled @ P_reverse.T + bias_reverse

best_model = joblib.load("rf_source_model.pkl")

y_pred = best_model.predict(transformed_target)
# print("Best Hyperparameters:", best_model.best_params_)
print("Validation Accuracy:", accuracy_score(y_fr, y_pred))
print("Classification Report:")
print(classification_report(y_fr, y_pred))


#testing source again
# y_pred = best_model.predict(X_en)
# # print("Best Hyperparameters:", best_model.best_params_)
# print("Validation Accuracy:", accuracy_score(y_en, y_pred))
# print("Classification Report:")
# print(classification_report(y_en, y_pred))