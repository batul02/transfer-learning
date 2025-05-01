from datasets import load_dataset
# from sentence_transformers import SentenceTransformer
import numpy as np

# dataset_en = load_dataset("xnli", "en", split="train")  # Source language: English
# dataset_fr = load_dataset("xnli", "fr", split="train")

# model = SentenceTransformer("sentence-transformers/nli-distilroberta-base-v2", device="cpu")

# def encode_sentence_pairs(dataset):
#     premises = dataset["premise"]
#     hypotheses = dataset["hypothesis"]
#     labels = dataset["label"]

#     premise_emb = model.encode(premises, batch_size=64, show_progress_bar=True)
#     hypo_emb = model.encode(hypotheses, batch_size=64, show_progress_bar=True)

#     # Concatenate embeddings
#     X = np.concatenate([premise_emb, hypo_emb], axis=1)
#     y = np.array(labels)
#     return X, y

# from sklearn.model_selection import StratifiedShuffleSplit

# Convert to list
# labels_en = dataset_en['label']
# labels_fr = dataset_fr['label']

# n_samples_en = 100000
# n_samples_fr = 30000

# split_en = StratifiedShuffleSplit(n_splits=1, test_size=n_samples_en, random_state=42)
# for _, subset_idx in split_en.split(X=[0]*len(labels_en), y=labels_en):
#     subset_en = dataset_en.select(subset_idx)

# # Stratified split for French
# split_fr = StratifiedShuffleSplit(n_splits=1, test_size=n_samples_fr, random_state=42)
# for _, subset_idx in split_fr.split(X=[0]*len(labels_fr), y=labels_fr):
#     subset_fr = dataset_fr.select(subset_idx)

# Select subsets of data for source (English) and target (French)
# subset_en = dataset_en.select(range(10000))  # Adjust the range as needed
# subset_fr = dataset_fr.select(range(5000))   # Adjust the range as needed

# Encode source and target domains
# X_en, y_en = encode_sentence_pairs(subset_en)
# X_fr, y_fr = encode_sentence_pairs(subset_fr)

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

# np.savez("encoded_data.npz", X_en=X_en, y_en=y_en, X_fr=X_fr, y_fr=y_fr)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import multiprocessing
import joblib

# Train-test split for the source domain
X_train, X_val, y_train, y_val = train_test_split(X_source_scaled, y_en, test_size=0.2, random_state=42)

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
n_jobs = min(64, multiprocessing.cpu_count())
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
random_search.fit(X_train, y_train)
# grid_search.fit(X_train, y_train)
joblib.dump(random_search, "random_search_rf_source.pkl")

# Best model
best_model = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_model.predict(X_val)
print("Best Hyperparameters:", random_search.best_params_)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred))

joblib.dump(best_model, 'rf_source_model.pkl')