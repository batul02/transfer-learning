import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import joblib
from scipy.stats import randint

data = torch.load("extracted_features.pt")
X_amazon = data["X_amazon_feats"]
y_amazon = data["y_amazon"]
X_dslr = data["X_dslr_feats"]
y_dslr = data["y_dslr"]

# Step 1: Split into train and test
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_dslr, y_dslr, test_size=0.4, stratify=y_dslr, random_state=42
)

np.savez("dslr_target_train_test_split.npz",
         X_target_train=X_target_train,
         X_target_test=X_target_test,
         y_target_train=y_target_train,
         y_target_test=y_target_test)

# Step 2: Set up Random Forest hyperparameter search
param_dist = {
    'n_estimators': randint(100, 800),
    'max_depth': [10, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Step 3: RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=50,  # Try 50 random combinations
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Step 4: Train
random_search.fit(X_target_train, y_target_train)

joblib.dump(random_search, "random_search_rf_dslr.pkl")

# Step 5: Best model evaluation
best_model = random_search.best_estimator_

y_pred = best_model.predict(X_target_test)
print("Best Hyperparameters:", random_search.best_params_)
print("Validation Accuracy:", accuracy_score(y_target_test, y_pred))
print("Classification Report:")
print(classification_report(y_target_test, y_pred))

# Step 6: Save best model
joblib.dump(best_model, 'best_target_model_dslr.pkl')
