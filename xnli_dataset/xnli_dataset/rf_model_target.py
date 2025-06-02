from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

data = np.load("encoded_data.npz")
X_en = data["X_en"]
y_en = data["y_en"]
X_fr = data["X_fr"]
y_fr = data["y_fr"]


print("Shape of Source Dataset: ", X_en.shape)
print("Shape of Target Dataset: ", X_fr.shape)

X_source_scaled = np.load("X_source_scaled.npy")
X_target_scaled = np.load("X_target_scaled.npy")

print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
print("Shape of Target Dataset scalled: ", X_target_scaled.shape)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import multiprocessing
import joblib

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target_scaled, y_fr, test_size=0.2, stratify=y_fr, random_state=42
)

np.savez("target_train_test_split.npz",
         X_target_train=X_target_train,
         X_target_test=X_target_test,
         y_target_train=y_target_train,
         y_target_test=y_target_test)

# Define hyperparameter grid for Random Forest
# param_grid = {
#     'n_estimators': [100, 200],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }
# param_grid = {
#     'n_estimators': [200, 500, 800],
#     'max_depth': [10, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None],
#     'bootstrap': [True, False]
# }

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 800),
    'max_depth': [10, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}


# Initialize Random Forest and GridSearch
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
# n_jobs = min(64, multiprocessing.cpu_count())
# grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs, verbose=2)
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,  # Try only 50 random combinations
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)
random_search.fit(X_target_train, y_target_train)

joblib.dump(random_search, "random_search_rf.pkl")

# grid_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_model.predict(X_target_test)
print("Best Hyperparameters:", random_search.best_params_)
print("Validation Accuracy:", accuracy_score(y_target_test, y_pred))
print("Classification Report:")
print(classification_report(y_target_test, y_pred))

joblib.dump(best_model, 'best_target_model.pkl')