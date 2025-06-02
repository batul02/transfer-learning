import numpy as np

X_source_scaled = np.load("X_source_scaled.npy")
# X_target_scaled = np.load("X_target_scaled.npy")

print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
# print("Shape of Target Dataset scalled: ", X_target_scaled.shape)

data = np.load("target_train_test_split.npz")
X_target_train = data["X_target_train"]
X_target_test = data["X_target_test"]
y_target_train = data["y_target_train"]
y_target_test = data["y_target_test"]

source_matrix = np.load("source_matrix.npy")
target_matrix = np.load("target_matrix.npy")

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1, fit_intercept=False)
ridge.fit(source_matrix, target_matrix)

P_matrix = ridge.coef_

if not(np.all(P_matrix == 0)):
    np.save("P_matrix_0.1.npy", P_matrix)
    print("saved")

