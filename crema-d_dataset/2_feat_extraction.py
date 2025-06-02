import pandas as pd
import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# Load the metadata
labels_df = pd.read_csv("crema_data.csv")

print("label df shape: ", labels_df.shape)

print(labels_df.head(5))


# Split into source (male) and target (female)
source_df = labels_df[labels_df['gender'] == 'male']
target_df = labels_df[labels_df['gender'] == 'female']

print("Shape of source df: ", len(source_df))
print("Shape of target df: ", len(target_df))

print(target_df.head(5))

# def extract_features(file_list, n_mfcc=40):
#     features = []
#     labels = []

#     for _, row in tqdm(file_list.iterrows(), total=len(file_list)):
#         path = row['path']
#         emotion = row['emotion']

#         try:
#             y, sr = librosa.load(path, sr=None)

#             # MFCCs and derivatives
#             mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#             mfcc_delta = librosa.feature.delta(mfcc)
#             mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

#             # Chroma
#             chroma = librosa.feature.chroma_stft(y=y, sr=sr)

#             # Mel spectrogram
#             mel = librosa.feature.melspectrogram(y=y, sr=sr)

#             # Tonnetz
#             tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

#             # Spectral features
#             spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#             spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#             spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#             spectral_flatness = librosa.feature.spectral_flatness(y=y)

#             # Other features
#             rms = librosa.feature.rms(y=y)
#             zcr = librosa.feature.zero_crossing_rate(y=y)

#             # Combine all features (mean + std for each)
#             combined_features = np.hstack([
#                 np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
#                 np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
#                 np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),
#                 np.mean(chroma, axis=1), np.std(chroma, axis=1),
#                 np.mean(mel, axis=1), np.std(mel, axis=1),
#                 np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
#                 np.mean(spectral_centroid), np.std(spectral_centroid),
#                 np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
#                 np.mean(spectral_contrast), np.std(spectral_contrast),
#                 np.mean(spectral_flatness), np.std(spectral_flatness),
#                 np.mean(rms), np.std(rms),
#                 np.mean(zcr), np.std(zcr)
#             ])

#             features.append(combined_features)
#             labels.append(emotion)

#         except Exception as e:
#             print(f"Failed to process {path}: {e}")

#     return np.array(features), np.array(labels)

def extract_features_parallel(df, n_mfcc=40, n_mels=128, fmin=50, fmax=8000, n_jobs=-1):
    """
    Parallel feature extraction for a DataFrame `df` with columns ['path','emotion'].
    Uses joblib to dispatch each file to a separate process/thread.
    Returns (X, y).
    """
    def process_row(path, emotion):
        try:
            y, sr = librosa.load(path, sr=None)
            
            # 1. MFCC + deltas
            mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_d  = librosa.feature.delta(mfcc)
            mfcc_dd = librosa.feature.delta(mfcc, order=2)
            
            # 2. Chroma
            chroma  = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # 3. Mel spectrogram
            mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                     fmin=fmin, fmax=fmax)
            # 4. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            
            # 5. Spectral features
            cent    = librosa.feature.spectral_centroid(y=y, sr=sr)
            bw      = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            flat    = librosa.feature.spectral_flatness(y=y)
            contrast= librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # 6. Energy + ZCR
            rms     = librosa.feature.rms(y=y)
            zcr     = librosa.feature.zero_crossing_rate(y=y)
            
            # 7. Pitch via pyin
            f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            pitch_stats = [np.mean(f0_clean) if len(f0_clean)>0 else 0,
                           np.std(f0_clean) if len(f0_clean)>0 else 0]
            
            # Aggregate stats: mean+std (and flatten arrays)
            def stats(x, axis):
                return [np.mean(x, axis=axis), np.std(x, axis=axis)]
            
            feature_vector = []
            # MFCC, delta, delta2
            for arr in (mfcc, mfcc_d, mfcc_dd):
                m, s = stats(arr, axis=1)
                feature_vector.extend(m); feature_vector.extend(s)
            # Chroma
            m, s = stats(chroma, axis=1); feature_vector.extend(m); feature_vector.extend(s)
            # Mel
            m, s = stats(mel, axis=1);    feature_vector.extend(m); feature_vector.extend(s)
            # Tonnetz
            m, s = stats(tonnetz, axis=1);feature_vector.extend(m); feature_vector.extend(s)
            # Spectral
            for arr in (cent, bw, flat):
                m, s = stats(arr, axis=1 if arr.ndim>1 else 0)
                feature_vector.extend(m if hasattr(m, '__iter__') else [m])
                feature_vector.extend(s if hasattr(s, '__iter__') else [s])
            # Contrast
            m, s = stats(contrast, axis=1); feature_vector.extend(m); feature_vector.extend(s)
            # RMS, ZCR
            m, s = stats(rms, axis=1); feature_vector.extend(m); feature_vector.extend(s)
            m, s = stats(zcr, axis=1); feature_vector.extend(m); feature_vector.extend(s)
            # Pitch
            feature_vector.extend(pitch_stats)
            # print("feature_vector", feature_vector)
            
            return np.array(feature_vector), emotion
        except Exception as e:
            # On failure, return None so we can filter it out
            return None, None

    # Launch in parallel
    # results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
    #     delayed(process_row)(row['path'], row['emotion'])
    #     for _, row in df.iterrows()
    # )

    tasks = ((row['path'], row['emotion']) for _, row in df.iterrows())
    total = len(df)

    with tqdm_joblib(tqdm(desc="Extracting features", total=total)) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_row)(path, emotion) for path, emotion in tasks
        )

    # Filter out failures and split features/labels
    feats, labs = zip(*[r for r in results if r[0] is not None])
    X = np.stack(feats)
    y = np.array(labs)
    return X, y

source_features, source_labels = extract_features_parallel(source_df)
target_features, target_labels = extract_features_parallel(target_df)

print("source_features shape: ", source_features.shape)
print("target_features shape: ", target_features.shape)

np.savez("crema_audio_librosa_features.npz",
         X_src=source_features, y_src=source_labels,
         X_tgt=target_features, y_tgt=target_labels)

