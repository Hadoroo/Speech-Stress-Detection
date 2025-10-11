import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from spafe.features.gfcc import gfcc
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# ====================== CONFIGURATIONS =======================
# ============================================================
dataset_folder = "Dataset/Processed"
features_dir = "Dataset/Acoustic_Features"
splits = ["train", "test"]
feat_types = ["MFCC", "GFCC", "LogFBank", "F0"]

# Buat folder output
for split in splits:
    for feat_type in feat_types:
        os.makedirs(os.path.join(features_dir, split, feat_type), exist_ok=True)

# ============================================================
# ==================== SPEC-AUGMENTATION =====================
# ============================================================
def spec_augment(mel_spectrogram, time_mask=20, freq_mask=8):
    mel = mel_spectrogram.copy()
    num_frames, num_mels = mel.shape

    # Time mask
    if num_frames > time_mask:
        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, num_frames - t)
        mel[t0:t0+t, :] = 0

    # Frequency mask
    if num_mels > freq_mask:
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, num_mels - f)
        mel[:, f0:f0+f] = 0

    return mel

# ============================================================
# ==================== FEATURE EXTRACTORS ====================
# ============================================================
def extract_features(file_path, sr=16000):
    """Ekstraksi MFCC, GFCC, LogFBank."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y = librosa.effects.preemphasis(y)
    
    # --- MFCC ---
    mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfcc_feat)
    mfcc_delta2 = librosa.feature.delta(mfcc_feat, order=2)
    mfcc_feat = np.vstack([mfcc_feat, mfcc_delta, mfcc_delta2])  # 39 dim

    # --- GFCC ---
    gfcc_feat = gfcc(
        sig=y,
        fs=sr,
        num_ceps=13,
        nfilts=40,
        nfft=2048,
        low_freq=0,
        high_freq=sr / 2
    )

    # --- LogFBank ---
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
    logfbank_feat = librosa.power_to_db(mel_spec, ref=np.max)
    logfbank_scaled = (logfbank_feat - np.min(logfbank_feat)) / (np.max(logfbank_feat) - np.min(logfbank_feat) + 1e-8)
    logfbank_scaled = spec_augment(logfbank_scaled)

    return mfcc_feat.T, gfcc_feat, logfbank_scaled.T


def extract_f0(file_path, sr=16000, fmin=50, fmax=500, frame_length=2048, hop_length=512):
    """Ekstraksi fundamental frequency (F0) berbasis FFT."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    f0_list = []

    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i:i + frame_length] * np.hamming(frame_length)

        # FFT
        spectrum = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(len(frame), 1 / sr)
        magnitude = np.abs(spectrum)

        valid_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if len(valid_idx) == 0:
            f0_list.append(0.0)
            continue

        peak_idx = valid_idx[np.argmax(magnitude[valid_idx])]
        f0 = freqs[peak_idx]
        f0_list.append(f0)

    return np.array(f0_list).reshape(-1, 1)

# ============================================================
# ======================= SCALER FITTING =====================
# ============================================================
scalers = {}
scaler_dir = "Results/Tuning/Acoustic_Features_Scaler"
scaler_paths = {ftype: os.path.join(scaler_dir, f"scaler_{ftype}.pkl") for ftype in ["MFCC", "GFCC", "LogFBank"]}
os.makedirs(scaler_dir, exist_ok=True)

if all(os.path.exists(path) for path in scaler_paths.values()):
    print("📂 Scaler ditemukan, langsung load...")
    for ftype in ["MFCC", "GFCC", "LogFBank"]:
        scalers[ftype] = joblib.load(scaler_paths[ftype])
else:
    print("🔎 Scaler belum ada, fitting dari data train...")
    all_train_feats = {ftype: [] for ftype in ["MFCC", "GFCC", "LogFBank"]}
    train_csv = "Dataset/CSV/train_split_stress.csv"
    df_train = pd.read_csv(train_csv)

    for _, row in tqdm(df_train.iterrows(), desc="Collecting train features"):
        filename = row["filename"]
        filepath = os.path.join(dataset_folder, filename)
        mfcc_feat, gfcc_feat, logfbank_feat = extract_features(filepath)
        all_train_feats["MFCC"].append(np.mean(mfcc_feat, axis=0))
        all_train_feats["GFCC"].append(np.mean(gfcc_feat, axis=0))
        all_train_feats["LogFBank"].append(np.mean(logfbank_feat, axis=0))

    for ftype in ["MFCC", "GFCC", "LogFBank"]:
        data = np.array(all_train_feats[ftype])
        scaler = StandardScaler().fit(data)
        scalers[ftype] = scaler
        joblib.dump(scaler, scaler_paths[ftype])
        print(f"✅ Scaler untuk {ftype} disimpan di {scaler_paths[ftype]}")

# ============================================================
# ======================= MAIN EXTRACTION ====================
# ============================================================
for split in splits:
    csv_path = f"Dataset/CSV/{split}_split_stress.csv"
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split} features"):
        filename = row["filename"]
        file_id = os.path.splitext(filename)[0]
        filepath = os.path.join(dataset_folder, filename)

        # Ekstraksi MFCC, GFCC, LogFBank
        mfcc_feat, gfcc_feat, logfbank_feat = extract_features(filepath)
        mfcc_scaled = scalers["MFCC"].transform(mfcc_feat.reshape(-1, mfcc_feat.shape[-1]))
        gfcc_scaled = scalers["GFCC"].transform(gfcc_feat.reshape(-1, gfcc_feat.shape[-1]))
        logfbank_scaled = scalers["LogFBank"].transform(logfbank_feat.reshape(-1, logfbank_feat.shape[-1]))

        np.save(os.path.join(features_dir, split, "MFCC", f"{file_id}_mfcc.npy"), mfcc_scaled)
        np.save(os.path.join(features_dir, split, "GFCC", f"{file_id}_gfcc.npy"), gfcc_scaled)
        np.save(os.path.join(features_dir, split, "LogFBank", f"{file_id}_logfbank.npy"), logfbank_scaled)

        # Ekstraksi F0 (tanpa normalisasi)
        f0_feat = extract_f0(filepath)
        np.save(os.path.join(features_dir, split, "F0", f"{file_id}_f0.npy"), f0_feat)

print("🎉 Semua fitur (MFCC, GFCC, LogFBank, F0) berhasil diekstraksi dan disimpan!")
print("📁 Output disimpan di:", features_dir)
