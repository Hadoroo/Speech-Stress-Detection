import os
import sys
import numpy as np
import librosa
from scipy.fft import dct
from tqdm import tqdm
import pandas as pd
import warnings

# Optional spafe (for gammatone filters)
try:
    from spafe.utils.filters import gammatone_filter_banks as spafe_gammatone_fb
    _HAS_SPAFE = True
except Exception:
    _HAS_SPAFE = False

# ==========================
# CONFIGURATIONS
# ==========================
SR = 16000
N_FFT = 2048
HOP = 512

# Baseline Kumar
MEL_N_MELS = 128

# Novelty features
MFCC_N_MELS = 64
LOGF_N_MELS = 80
GFCC_NFILTS = 48
NUM_CEPS = 13

USE_CMVN = True

datasets = ["CREMAD"]
splits = ["train", "test"]
feat_types = ["MEL", "MFCC", "GFCC", "LogFBank"]

# ==========================
# UTILITIES
# ==========================
def preemphasis(x, c=0.97):
    x = x.astype(np.float32)
    return np.append(x[0], x[1:] - c * x[:-1]).astype(np.float32)

def pad_if_short(y, target_len=N_FFT):
    """Pad audio jika lebih pendek dari frame_length (N_FFT)."""
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
    return y

def framing(y, n_fft=N_FFT, hop=HOP):
    """Return (T, n_fft) minimal 1 frame (di-pad jika perlu)."""
    y = pad_if_short(y, n_fft)
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop).T

    if frames.shape[0] == 0:
        # Force 1 frame padded with zeros
        frames = np.zeros((1, n_fft), dtype=np.float32)

    win = np.hamming(n_fft).astype(np.float32)
    return frames * win

def power_spectrum(frames):
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    return (1.0 / N_FFT) * (np.abs(spec) ** 2 + 1e-12)

def cmvn(feat):
    mean = np.mean(feat, axis=0, keepdims=True)
    std = np.std(feat, axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    return ((feat - mean) / std).astype(np.float32)

_CACHE = {}

def mel_fbanks(sr, n_fft, n_mels):
    key = ("mel", sr, n_fft, n_mels)
    if key not in _CACHE:
        _CACHE[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)
    return _CACHE[key]

def gammatone_fbanks(sr, n_fft, nfilts):
    key = ("gt", sr, n_fft, nfilts)
    if key in _CACHE:
        return _CACHE[key]

    n_bins = n_fft // 2 + 1

    if _HAS_SPAFE:
        gt = spafe_gammatone_fb(nfilts=nfilts, nfft=n_fft, fs=sr)
        gt = np.asarray(gt, dtype=np.float32)
    else:
        # Deterministic fallback
        mel = librosa.core.mel_frequencies(n_mels=nfilts, fmin=0.0, fmax=sr/2.0)
        freqs = np.linspace(0, sr/2.0, n_bins)
        gt = np.zeros((nfilts, n_bins), dtype=np.float32)
        for i, c in enumerate(mel):
            width = max(50.0, c * 0.25 + 50.0)
            left, right = c - width, c + width
            left_mask = (freqs >= left) & (freqs <= c)
            right_mask = (freqs > c) & (freqs <= right)
            if left_mask.any():
                gt[i, left_mask] = (freqs[left_mask] - left) / (c - left + 1e-12)
            if right_mask.any():
                gt[i, right_mask] = (right - freqs[right_mask]) / (right - c + 1e-12)
            gt[i] /= (gt[i].sum() + 1e-12)

    _CACHE[key] = gt.astype(np.float32)
    return _CACHE[key]

def sanitize_feat(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features(filepath):
    try:
        y, _ = librosa.load(filepath, sr=SR, mono=True)
    except Exception:
        return None

    # Always pad if needed → no skipping
    y = pad_if_short(y, N_FFT)
    y = preemphasis(y)
    frames = framing(y)
    power = power_spectrum(frames)

    # ===== MEL =====
    mel_fb = mel_fbanks(SR, N_FFT, MEL_N_MELS)
    mel = np.dot(power, mel_fb.T)
    mel = np.log(np.maximum(mel, 1e-12))
    if USE_CMVN:
        mel = cmvn(mel)
    mel = sanitize_feat(mel)

    # ===== MFCC =====
    mel_fb_mfcc = mel_fbanks(SR, N_FFT, MFCC_N_MELS)
    mel_energy = np.dot(power, mel_fb_mfcc.T)
    mel_energy = np.log(np.maximum(mel_energy, 1e-12))
    mfcc = dct(mel_energy, type=2, axis=1, norm="ortho")[:, :NUM_CEPS]
    mfcc_T = mfcc.T
    mfcc_d1 = librosa.feature.delta(mfcc_T, order=1).T
    mfcc_d2 = librosa.feature.delta(mfcc_T, order=2).T
    mfcc_full = np.concatenate([mfcc, mfcc_d1, mfcc_d2], axis=1)
    if USE_CMVN:
        mfcc_full = cmvn(mfcc_full)
    mfcc_full = sanitize_feat(mfcc_full)

    # ===== GFCC =====
    gt_fb = gammatone_fbanks(SR, N_FFT, GFCC_NFILTS)
    gt_energy = np.dot(power, gt_fb.T)
    gt_energy = np.maximum(gt_energy, 1e-12)
    gfcc = dct(np.log(gt_energy), type=2, axis=1, norm="ortho")[:, :NUM_CEPS]
    gfcc_T = gfcc.T
    gfcc_d1 = librosa.feature.delta(gfcc_T, order=1).T
    gfcc_d2 = librosa.feature.delta(gfcc_T, order=2).T
    gfcc_full = np.concatenate([gfcc, gfcc_d1, gfcc_d2], axis=1)
    if USE_CMVN:
        gfcc_full = cmvn(gfcc_full)
    gfcc_full = sanitize_feat(gfcc_full)

    # ===== LogFBank =====
    mel_fb_log = mel_fbanks(SR, N_FFT, LOGF_N_MELS)
    logf = np.dot(power, mel_fb_log.T)
    logf = np.log(np.maximum(logf, 1e-12))
    if USE_CMVN:
        logf = cmvn(logf)
    logf = sanitize_feat(logf)

    return mel.astype(np.float32), mfcc_full.astype(np.float32), gfcc_full.astype(np.float32), logf.astype(np.float32)


# ==========================
# MAIN PIPELINE
# ==========================
def main():
    for dataset in datasets:
        print(f"\nExtracting features for dataset: {dataset}")

        processed = f"Dataset/{dataset}/Processed"
        csv_dir = f"Dataset/{dataset}/CSV"
        out_dir = f"Dataset/{dataset}/Acoustic_Features"

        for split in splits:
            for ft in feat_types:
                os.makedirs(os.path.join(out_dir, split, ft), exist_ok=True)

        for split in splits:
            csv_path = os.path.join(csv_dir, f"{split}_split_stress.csv")
            if not os.path.exists(csv_path):
                print("CSV missing:", csv_path)
                continue

            df = pd.read_csv(csv_path)

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset}-{split}"):
                fname = row["filename"]
                id_ = os.path.splitext(fname)[0]
                path = os.path.join(processed, fname)

                res = extract_features(path)
                if res is None:
                    continue

                mel, mfcc_feat, gfcc_feat, logf_feat = res

                np.save(os.path.join(out_dir, split, "MEL", f"{id_}_mel.npy"), mel)
                np.save(os.path.join(out_dir, split, "MFCC", f"{id_}_mfcc.npy"), mfcc_feat)
                np.save(os.path.join(out_dir, split, "GFCC", f"{id_}_gfcc.npy"), gfcc_feat)
                np.save(os.path.join(out_dir, split, "LogFBank", f"{id_}_logfbank.npy"), logf_feat)

    print("\n🎉 Feature extraction completed.")


if __name__ == "__main__":
    main()
