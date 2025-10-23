import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from spafe.features.gfcc import gfcc

# ============================================================
# ====================== CONFIGURATIONS =======================
# ============================================================
datasets = ["RAVDESS", "TESS", "CREMAD"]
splits = ["train", "test"]
feat_types = ["MFCC", "GFCC", "LogFBank", "F0"]

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
# ======================= MAIN EXTRACTION ====================
# ============================================================
for dataset_name in datasets:
    print(f"\n📁 Memproses dataset: {dataset_name}")

    processed_folder = f"Dataset/{dataset_name}/Processed"
    csv_folder = f"Dataset/{dataset_name}/CSV"
    features_dir = f"Dataset/{dataset_name}/Acoustic_Features"

    # Buat struktur folder per dataset
    for split in splits:
        for feat_type in feat_types:
            os.makedirs(os.path.join(features_dir, split, feat_type), exist_ok=True)

    for split in splits:
        csv_path = os.path.join(csv_folder, f"{split}_split_stress.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ CSV {csv_path} tidak ditemukan, dilewati.")
            continue

        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_name} - {split}"):
            filename = row["filename"]
            file_id = os.path.splitext(filename)[0]
            filepath = os.path.join(processed_folder, filename)

            if not os.path.exists(filepath):
                continue

            # Ekstraksi MFCC, GFCC, LogFBank
            mfcc_feat, gfcc_feat, logfbank_feat = extract_features(filepath)
            np.save(os.path.join(features_dir, split, "MFCC", f"{file_id}_mfcc.npy"), mfcc_feat)
            np.save(os.path.join(features_dir, split, "GFCC", f"{file_id}_gfcc.npy"), gfcc_feat)
            np.save(os.path.join(features_dir, split, "LogFBank", f"{file_id}_logfbank.npy"), logfbank_feat)

            # Ekstraksi F0 (tanpa normalisasi)
            f0_feat = extract_f0(filepath)
            np.save(os.path.join(features_dir, split, "F0", f"{file_id}_f0.npy"), f0_feat)

print("🎉 Semua fitur (MFCC, GFCC, LogFBank, F0) berhasil diekstraksi tanpa standarisasi!")
print("📁 Output disimpan di masing-masing folder Dataset/<dataset_name>/Acoustic_Features/")
