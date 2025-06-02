import numpy as np
import gc

data = np.load("encoded_data.npz")
X_en = data["X_en"]
y_en = data["y_en"]
X_fr = data["X_fr"]
y_fr = data["y_fr"]

print("Shape of Source Dataset: ", X_en.shape)
print("Shape of Target Dataset: ", X_fr.shape)

from sklearn.preprocessing import StandardScaler
import numpy as np

n_models = max(X_en.shape[1], X_fr.shape[1])  # Number of models for bagging
source_matrices = []
target_matrices = []

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_en)
X_target_scaled = scaler.transform(X_fr)

# scaler = StandardScaler()
# X_combined = np.vstack((X_en, X_fr))
# X_combined_scaled = scaler.fit_transform(X_combined)
# X_source_scaled = X_combined_scaled[:len(X_en)]
# X_target_scaled = X_combined_scaled[len(X_en):]

print("Shape of Source Dataset scalled: ", X_source_scaled.shape)
print("Shape of Target Dataset scalled: ", X_target_scaled.shape)

np.save("X_source_scaled.npy", X_source_scaled)
np.save("X_target_scaled.npy", X_target_scaled)

def sample_by_class_distribution(X, y, label_ratios, total_samples=None, seed=42):
    X_sampled, y_sampled = [], []
    labels = np.unique(y)
    total = total_samples or len(y)

    for label, ratio in zip(labels, label_ratios):
        X_label = X[y == label]
        y_label = y[y == label]
        sample_size = min(int(ratio * total), len(X_label))
        X_l, y_l = resample(X_label, y_label, n_samples=sample_size, random_state=seed)

        # idx = np.random.permutation(len(X_l))
        # X_l, y_l = X_l[idx], y_l[idx]

        X_sampled.append(X_l)
        y_sampled.append(y_l)

    return np.vstack(X_sampled), np.hstack(y_sampled)

import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# def run_model_trials(i, X_source_scaled, y_source, X_target_scaled, y_target, n_trials, n_classes):

#     random_counts = np.random.randint(50, 500, size=n_classes)
#     label_ratios = random_counts / np.sum(random_counts)

#     source_trials = []
#     target_trials = []

#     for trial in range(n_trials):
#         trial_seed = i * 100 + trial

#         # Target
#         X_target_bagged, y_target_bagged = sample_by_class_distribution(
#             X_target_scaled, y_target, label_ratios, total_samples=len(y_target), seed=trial_seed
#         )
#         # print(X_target_bagged.shape, y_target_bagged[0])
#         model_target = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=32)
#         model_target.fit(X_target_bagged, y_target_bagged)
#         target_trials.append(model_target.feature_importances_)
#         del model_target
#         gc.collect()

#         # Source
#         X_source_bagged, y_source_bagged = sample_by_class_distribution(
#             X_source_scaled, y_source, label_ratios, total_samples=len(y_source), seed=trial_seed
#         )
#         model_source = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=32)
#         model_source.fit(X_source_bagged, y_source_bagged)
#         source_trials.append(model_source.feature_importances_)
#         del model_source
#         gc.collect()

#     if i % 10 == 0:
#         print(f"Iteration {i}/{n_models}")

#     return np.mean(source_trials, axis=0), np.mean(target_trials, axis=0)


import gc
from sklearn.ensemble import RandomForestClassifier

def run_model_once(i, X_source_scaled, y_source, X_target_scaled, y_target, n_classes):
    # Generate class distribution
    random_counts = np.random.randint(50, 500, size=n_classes)
    label_ratios = random_counts / np.sum(random_counts)
    trial_seed = i  # use i as seed

    # Target
    X_target_bagged, y_target_bagged = sample_by_class_distribution(
        X_target_scaled, y_target, label_ratios, total_samples=len(y_target), seed=trial_seed
    )
    model_target = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=-1)
    model_target.fit(X_target_bagged, y_target_bagged)
    target_importance = model_target.feature_importances_
    del model_target
    gc.collect()

    # Source
    X_source_bagged, y_source_bagged = sample_by_class_distribution(
        X_source_scaled, y_source, label_ratios, total_samples=len(y_source), seed=trial_seed
    )
    model_source = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=-1)
    model_source.fit(X_source_bagged, y_source_bagged)
    source_importance = model_source.feature_importances_
    del model_source
    gc.collect()

    if i % 10 == 0:
        print(f"Iteration {i}/{n_models}")

    return source_importance, target_importance


import multiprocessing
from sklearn.utils import resample
from joblib import Parallel, delayed

# n_models = 10
n_trials = 1
n_classes = len(np.unique(y_fr))

n_jobs = min(64, multiprocessing.cpu_count())
print("multiprocessing.cpu_count()", multiprocessing.cpu_count())
# results = Parallel(n_jobs=n_jobs)(
#     delayed(run_model_trials)(i, X_source_scaled, y_en, X_target_scaled, y_fr, n_trials, n_classes)
#     for i in range(n_models)
# )
results = Parallel(n_jobs=n_jobs)(
    delayed(run_model_once)(i, X_source_scaled, y_en, X_target_scaled, y_fr, n_classes)
    for i in range(n_models)
)

source_matrices, target_matrices = zip(*results)
source_matrix = np.array(source_matrices)
target_matrix = np.array(target_matrices)

print(np.all(source_matrix == 0))
print(np.all(target_matrix == 0))

np.save("source_matrix.npy", source_matrix)
np.save("target_matrix.npy", target_matrix)

#Feature-level discrepancy
feature_discrepancy = np.abs(source_matrix.mean(axis=0) - target_matrix.mean(axis=0))
mean_discrepancy = feature_discrepancy.mean()
print("Mean Feature Discrepancy:", mean_discrepancy)


# Cosine Similarity (between feature means)
from sklearn.metrics.pairwise import cosine_similarity

source_mean = source_matrix.mean(axis=0).reshape(1, -1)
target_mean = target_matrix.mean(axis=0).reshape(1, -1)

cos_sim = cosine_similarity(source_mean, target_mean)[0][0]
print("Cosine Similarity:", cos_sim)

#Correlation (Pearson)
from scipy.stats import pearsonr

corr, _ = pearsonr(source_mean.flatten(), target_mean.flatten())
print("Pearson Correlation:", corr)

#KL Divergence (after normalizing to distributions)
from scipy.special import rel_entr

# Normalize mean vectors to sum to 1 (like probabilities)
p = source_mean.flatten()
q = target_mean.flatten()
p /= p.sum()
q /= q.sum()

kl_div = np.sum(rel_entr(p, q))
print("KL Divergence (source || target):", kl_div)