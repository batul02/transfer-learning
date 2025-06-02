import logging
import random
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import joblib

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

data = np.load("crema_audio_librosa_features.npz")

# Extract arrays
X_src = data["X_src"]   # Source feature matrix
y_src = data["y_src"]   # Source labels

X_tgt = data["X_tgt"]   # Target feature matrix
y_tgt = data["y_tgt"]

X_tgt = np.array(X_tgt)
y_tgt = np.array(y_tgt)

# Get unique classes and their samples
downsampled_X = []
downsampled_y = []

for emotion in np.unique(y_tgt):
    # Get indices of current emotion
    indices = np.where(y_tgt == emotion)[0]
    
    # Downsample 50% (with fixed seed)
    selected = resample(indices, replace=False, 
                        n_samples=len(indices) // 2, random_state=42)
    
    downsampled_X.append(X_tgt[selected])
    downsampled_y.append(y_tgt[selected])

# Stack everything
X_tgt_downsampled = np.vstack(downsampled_X)
y_tgt_downsampled = np.hstack(downsampled_y)

print("Downsampled target shape:", X_tgt_downsampled.shape, y_tgt_downsampled.shape)

np.savez("crema_audio_librosa_features_downsized_trg.npz",
         X_src=X_src, y_src=y_src,
         X_tgt=X_tgt_downsampled, y_tgt=y_tgt_downsampled)


scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_src)
X_target_scaled = scaler.transform(X_tgt_downsampled)


y_source = y_src
y_target = y_tgt_downsampled

le = LabelEncoder()
y_source = le.fit_transform(y_source)
y_target = le.transform(y_target)

joblib.dump(le, "label_encoder.pkl")

np.savez("crema_audio_librosa_features_downsized_trg_label_enc.npz",
         X_src=X_src, y_src=y_source,
         X_tgt=X_tgt_downsampled, y_tgt=y_target)

n_models = 4000 
n_trials = 3
n_classes = len(np.unique(y_target))

# def sample_by_class_distribution(X, y, label_ratios, total_samples=None, seed=42):
#     X_sampled, y_sampled = [], []
#     labels = np.unique(y)
#     total = total_samples or len(y)

#     for label, ratio in zip(labels, label_ratios):
#         X_label = X[y == label]
#         y_label = y[y == label]
#         sample_size = min(int(ratio * total), len(X_label))
#         X_l, y_l = resample(X_label, y_label, n_samples=sample_size, random_state=seed)
#         X_sampled.append(X_l)
#         y_sampled.append(y_l)

#     return np.vstack(X_sampled), np.hstack(y_sampled)

def sample_by_class_distribution(X, y, label_ratios, total_samples=None, seed=42):
    rng = np.random.default_rng(seed)
    X_sampled, y_sampled = [], []
    labels = np.unique(y)
    total = total_samples or len(y)

    for label, ratio in zip(labels, label_ratios):
        idx = np.where(y == label)[0]
        sample_size = min(int(ratio * total), len(idx))
        sampled_idx = rng.choice(idx, size=sample_size, replace=True)
        X_sampled.append(X[sampled_idx])
        y_sampled.append(y[sampled_idx])

    return np.vstack(X_sampled), np.hstack(y_sampled)

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
#         model_target = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=-1)
#         model_target.fit(X_target_bagged, y_target_bagged)
#         target_trials.append(model_target.feature_importances_)

#         # Source
#         X_source_bagged, y_source_bagged = sample_by_class_distribution(
#             X_source_scaled, y_source, label_ratios, total_samples=len(y_source), seed=trial_seed
#         )
#         model_source = RandomForestClassifier(n_estimators=100, random_state=trial_seed, n_jobs=-1)
#         model_source.fit(X_source_bagged, y_source_bagged)
#         source_trials.append(model_source.feature_importances_)

#     if i % 10 == 0:
#         print(f"Iteration {i}/{n_models}")

#     return np.mean(source_trials, axis=0), np.mean(target_trials, axis=0)

def run_model_trials(i, X_source_scaled, y_source, X_target_scaled, y_target, n_trials, n_classes):
    rng = np.random.default_rng(seed=i)
    random_counts = rng.integers(50, 500, size=n_classes)
    label_ratios = random_counts / np.sum(random_counts)
    trial_seeds = rng.integers(0, 1e6, size=n_trials)

    source_trials = []
    target_trials = []

    for trial in range(n_trials):
        seed = trial_seeds[trial]

        # Target
        X_target_bagged, y_target_bagged = sample_by_class_distribution(
            X_target_scaled, y_target, label_ratios, total_samples=len(y_target), seed=seed
        )
        model_target = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        model_target.fit(X_target_bagged, y_target_bagged)
        target_trials.append(model_target.feature_importances_.astype(np.float32))

        # Source
        X_source_bagged, y_source_bagged = sample_by_class_distribution(
            X_source_scaled, y_source, label_ratios, total_samples=len(y_source), seed=seed
        )
        model_source = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        model_source.fit(X_source_bagged, y_source_bagged)
        source_trials.append(model_source.feature_importances_.astype(np.float32))

    if i % 10 == 0:
        logging.info(f"Iteration {i}/{n_models}")

    return np.mean(source_trials, axis=0), np.mean(target_trials, axis=0)


n_jobs = -1

# results = Parallel(n_jobs=n_jobs)(
#     delayed(run_model_trials)(i, X_source_scaled, y_source, X_target_scaled, y_target, n_trials, n_classes)
#     for i in range(n_models)
# )

results = Parallel(n_jobs=n_jobs)(
    delayed(run_model_trials)(i, X_source_scaled, y_source, X_target_scaled, y_target, n_trials, n_classes)
    for i in tqdm(range(n_models))
)

source_matrices, target_matrices = zip(*results)
source_matrix = np.array(source_matrices)
target_matrix = np.array(target_matrices)

torch.save({
    "source_matrix": source_matrix,
    "target_matrix": target_matrix,
}, "cremad_feature_correspondence.pt")

print("Saved source and target matrices!")

