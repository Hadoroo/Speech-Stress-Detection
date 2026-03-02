import os
import numpy as np
import librosa
from scipy.signal import lfilter
from spafe.features.gfcc import gfcc
from tqdm import tqdm

# =====================
# CONFIG
# =====================
SR = 16000
N_MFCC = 40
N_GFCC = 40
N_LPC = 16
N_FFT = 2048
HOP = 512
MAX_LEN = 128

INPUT_DIR = "Dataset/CREMAD/Processed"
OUTPUT_DIR = "Dataset/CREMAD/Acoustic_Features"

# =====================
# UTILS
# =====================
def pad_or_truncate(feat, max_len):
    if feat.shape[1] < max_len:
        pad = max_len - feat.shape[1]
        return np.pad(feat, ((0,0),(0,pad)))
    return feat[:, :max_len]

def normalize(feat, eps=1e-8):
    mean = np.mean(feat, axis=1, keepdims=True)
    std = np.std(feat, axis=1, keepdims=True)
    return (feat - mean) / (std + eps)

# =====================
# FEATURE EXTRACTION
# =====================
def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP
    )

    mfcc = normalize(mfcc)

    return pad_or_truncate(mfcc, MAX_LEN)

def extract_gfcc(y):
    g = gfcc(
        sig=y,
        fs=SR,
        num_ceps=N_GFCC,
        nfilts=64,
        nfft=N_FFT
    ).T

    g = normalize(g)

    return pad_or_truncate(g, MAX_LEN)


def extract_lpc(y):
    lpc_feat = []

    frames = librosa.util.frame(
        y,
        frame_length=N_FFT,
        hop_length=HOP
    )

    for frame in frames.T:
        coeff = librosa.lpc(frame, order=N_LPC)
        lpc_feat.append(coeff[1:])

    lpc_feat = np.array(lpc_feat).T
    lpc_feat = normalize(lpc_feat)

    return pad_or_truncate(lpc_feat, MAX_LEN)


# =====================
# MAIN PIPELINE
# =====================
for root, _, files in os.walk(INPUT_DIR):
    for file in tqdm(files, desc="Processing audio"):
        if not file.endswith(".wav"):
            continue

        audio_path = os.path.join(root, file)
        rel_path = os.path.relpath(root, INPUT_DIR)

        # load audio
        y, _ = librosa.load(audio_path, sr=SR)

        mfcc = extract_mfcc(y)
        gfcc_feat = extract_gfcc(y)
        lpc = extract_lpc(y)

        # output dirs
        for feat, name in zip(
            [mfcc, gfcc_feat, lpc],
            ["MFCC", "GFCC", "LPC"]
        ):
            out_dir = os.path.join(OUTPUT_DIR, name, rel_path)
            os.makedirs(out_dir, exist_ok=True)

            np.save(
                os.path.join(out_dir, file.replace(".wav", ".npy")),
                feat
            )

print("✅ Semua fitur berhasil diekstrak.")

