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

data = np.load("../crema_audio_librosa_features.npz")

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



scaler = StandardScaler().fit(X_src)
X_src_sc = scaler.transform(X_src)
X_tgt_sc = scaler.transform(X_tgt_downsampled)


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
n_trials = 5
n_classes = len(np.unique(y_target))


from sklearn.metrics.pairwise import cosine_similarity

paired_src = []
paired_tgt = []
for cls in np.unique(y_target):
    src_idx = np.where(y_source == cls)[0]
    tgt_idx = np.where(y_target == cls)[0]
    S = X_src_sc[src_idx]
    T = X_tgt_sc[tgt_idx]
    sim = cosine_similarity(T, S)           # shape (|T|,|S|)
    best_src = sim.argmax(axis=1)           # which S best matches each T
    for ti, si in enumerate(best_src):
        paired_src.append(S[si])
        paired_tgt.append(T[ti])

source_matrix = np.vstack(paired_src)       # now shape (N_pairs, n_features)
target_matrix = np.vstack(paired_tgt)

print("Shape of source_matrix", source_matrix.shape)
print("Shape of target_matrix", target_matrix.shape)

torch.save({
    "source_matrix": source_matrix,
    "target_matrix": target_matrix,
}, "cremad_feature_correspondence.pt")

print("Saved source and target matrices!")

