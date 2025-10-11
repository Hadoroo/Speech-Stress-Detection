import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random

input_folder = "Dataset/Raw"
output_folder = "Dataset/Processed"
os.makedirs(output_folder, exist_ok=True)

def augment_audio(y, sr):
    """Augmentasi audio sederhana dan kompatibel dengan semua versi librosa"""
    augmentations = []
    
    # 1. Additive noise (SNR 10–20 dB)
    noise = np.random.randn(len(y))
    snr_db = random.uniform(10, 20)
    y_noisy = y + noise * (np.linalg.norm(y) / (10**(snr_db/20) * np.linalg.norm(noise)))
    augmentations.append(y_noisy)

    # 2. Time shift ±0.1s
    shift = int(sr * random.uniform(-0.1, 0.1))
    augmentations.append(np.roll(y, shift))

    # 3. Pitch shift ±2 semitone
    n_steps = random.uniform(-2, 2)
    augmentations.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps))

    # 4. Speed perturbation ±10% (versi waveform)
    rate = random.uniform(0.9, 1.1)
    new_len = int(len(y) / rate)
    y_stretch = librosa.resample(y, orig_sr=sr, target_sr=int(sr * rate))
    # jika hasil lebih panjang/pendek, potong/pad
    if len(y_stretch) > len(y):
        y_stretch = y_stretch[:len(y)]
    else:
        y_stretch = np.pad(y_stretch, (0, len(y) - len(y_stretch)))
    augmentations.append(y_stretch)

    return augmentations


def process_audio(file_path, out_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Normalisasi RMS ke target dBFS
    y = librosa.util.normalize(y)

    # Simpan versi asli
    sf.write(out_path, y, sr, subtype="PCM_16")

    # Buat 3 versi augmentasi acak
    for i, y_aug in enumerate(augment_audio(y, sr)):
        aug_path = out_path.replace(".wav", f"_aug{i+1}.wav")
        sf.write(aug_path, y_aug, sr, subtype="PCM_16")

files = [f for f in os.listdir(input_folder) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]

for file in tqdm(files, desc="Processing audio files", unit="file"):
    in_path = os.path.join(input_folder, file)
    out_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".wav")
    process_audio(in_path, out_path)

print("✅ Selesai! Semua file dan augmentasi disimpan di:", output_folder)
