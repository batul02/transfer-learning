import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import numpy as np
from sklearn.utils import resample
import joblib
from collections import Counter

data = torch.load("extracted_features.pt")
X_amazon = data["X_amazon_feats"]
y_amazon = data["y_amazon"]
X_dslr = data["X_dslr_feats"]
y_dslr = data["y_dslr"]

y_amazon_np = y_amazon.numpy() if isinstance(y_amazon, torch.Tensor) else np.array(y_amazon)
y_dslr_np = y_dslr.numpy() if isinstance(y_dslr, torch.Tensor) else np.array(y_dslr)

# Shapes
print("Amazon features shape:", X_amazon.shape)
print("Amazon labels shape:", y_amazon.shape)
print("DSLR features shape:", X_dslr.shape)
print("DSLR labels shape:", y_dslr.shape)

# Unique labels
print("Amazon unique labels:", np.unique(y_amazon_np))
print("DSLR unique labels:", np.unique(y_dslr_np))

# Samples per label
print("Amazon label distribution:", dict(Counter(y_amazon_np)))
print("DSLR label distribution:", dict(Counter(y_dslr_np)))

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_amazon)
X_target_scaled = scaler.transform(X_dslr)

y_source = y_amazon
y_target = y_dslr

le = LabelEncoder()
y_source = le.fit_transform(y_source)
y_target = le.transform(y_target)

joblib.dump(le, "label_encoder.pkl")

np.savez("extracted_features_label_enc_feat_scl.npz",
         X_src=X_source_scaled, y_src=y_source,
         X_tgt=X_target_scaled, y_tgt=y_target)

# n_models = max(X_source_scaled.shape[1], X_target_scaled.shape[1])
n_models = 4000
n_trials = 5
n_classes = len(np.unique(y_target))

def sample_by_class_distribution(X, y, label_ratios, total_samples=None, seed=42):
    X_sampled, y_sampled = [], []
    labels = np.unique(y)
    total = total_samples or len(y)

    for label, ratio in zip(labels, label_ratios):
        X_label = X[y == label]
        y_label = y[y == label]
        sample_size = min(int(ratio * total), len(X_label))
        X_l, y_l = resample(X_label, y_label, n_samples=sample_size, random_state=seed)
        X_sampled.append(X_l)
        y_sampled.append(y_l)

    return np.vstack(X_sampled), np.hstack(y_sampled)

def run_model_trials(i, X_source_scaled, y_source, X_target_scaled, y_target, n_trials, n_classes):

    random_counts = np.random.randint(50, 500, size=n_classes)
    label_ratios = random_counts / np.sum(random_counts)

    source_trials = []
    target_trials = []

    for trial in range(n_trials):
        trial_seed = i * 100 + trial

        # Target
        X_target_bagged, y_target_bagged = sample_by_class_distribution(
            X_target_scaled, y_target, label_ratios, total_samples=len(y_target), seed=trial_seed
        )
        model_target = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=-1)
        model_target.fit(X_target_bagged, y_target_bagged)
        target_trials.append(model_target.feature_importances_)

        # Source
        X_source_bagged, y_source_bagged = sample_by_class_distribution(
            X_source_scaled, y_source, label_ratios, total_samples=len(y_source), seed=trial_seed
        )
        model_source = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=-1)
        model_source.fit(X_source_bagged, y_source_bagged)
        source_trials.append(model_source.feature_importances_)

    if i % 10 == 0:
        print(f"Iteration {i}/{n_models}")

    return np.mean(source_trials, axis=0), np.mean(target_trials, axis=0)

n_jobs = -1

results = Parallel(n_jobs=n_jobs)(
    delayed(run_model_trials)(i, X_source_scaled, y_source, X_target_scaled, y_target, n_trials, n_classes)
    for i in range(n_models)
)

source_matrices, target_matrices = zip(*results)
source_matrix = np.array(source_matrices)
target_matrix = np.array(target_matrices)

torch.save({
    "source_matrix": source_matrix,
    "target_matrix": target_matrix,
}, "office31_feature_correspondence.pt")

print("Saved source and target matrices!")

