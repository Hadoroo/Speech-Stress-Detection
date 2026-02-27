import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

Dataset = ['CREMAD']

TARGET_SR = 16000
TARGET_PEAK = 10 ** (-1 / 20)   # -1 dBFS ≈ 0.891

def peak_normalize(y, peak_target=TARGET_PEAK):
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    return y * (peak_target / peak)

for dataset in Dataset:
    input_folder = f"Dataset/{dataset}/Raw"
    output_folder = f"Dataset/{dataset}/Processed"
    os.makedirs(output_folder, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    ]

    for file in tqdm(files, desc=f"Processing {dataset}", unit="file"):
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(
            output_folder, os.path.splitext(file)[0] + ".wav"
        )

        # Load audio → 16 kHz → mono
        y, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

        # Peak normalize to -1 dBFS
        y = peak_normalize(y, TARGET_PEAK)

        # Save as 16-bit PCM WAV
        sf.write(out_path, y, TARGET_SR, subtype="PCM_16")

print("Selesai")
