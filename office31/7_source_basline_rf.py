import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

data = np.load("extracted_features_label_enc_feat_scl.npz")

# Extract arrays
X_amazon = data["X_src"]   # Source feature matrix
y_amazon = data["y_src"]   # Source labels

X_dslr = data["X_tgt"]   # Target feature matrix
y_dslr = data["y_tgt"]

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_amazon, y_amazon, test_size=0.2, stratify=y_amazon, random_state=42
)

print("\nTraining baseline model on Source...")

param_dist = {
    'n_estimators': randint(100, 800),
    'max_depth': [10, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_source_train, y_source_train)
baseline_model = random_search.best_estimator_
le = joblib.load("label_encoder.pkl")

# Save baseline model
# joblib.dump(baseline_model, "random_search_rf_amazon.pkl")

# Evaluate baseline
y_pred_source = baseline_model.predict(X_source_test)
y_pred_source = le.inverse_transform(y_pred_source)
y_source_test = le.inverse_transform(y_source_test)
print("\nBaseline model performance on source test set:")
print("Accuracy:", accuracy_score(y_source_test, y_pred_source))
print(classification_report(y_source_test, y_pred_source))

joblib.dump(baseline_model, 'best_source_model_dslr.pkl')