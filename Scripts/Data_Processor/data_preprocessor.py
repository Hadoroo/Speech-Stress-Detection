import os
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

Dataset = ['RAVDESS', 'CREMAD', 'TESS']

for dataset in Dataset:
    input_folder = f"Dataset/{dataset}/Raw"
    output_folder = f"Dataset/{dataset}/Processed"
    os.makedirs(output_folder, exist_ok=True)

    def process_audio(file_path, out_path):
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        # Optional: trim silence jika perlu
        # y, _ = librosa.effects.trim(y, top_db=20)

        # Normalisasi RMS ke target dBFS
        y = librosa.util.normalize(y)

        # Simpan versi asli saja (tanpa augmentasi)
        sf.write(out_path, y, sr, subtype="PCM_16")

    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]

    for file in tqdm(files, desc=f"Processing {dataset}", unit="file"):
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, os.path.splitext(file)[0] + ".wav")
        process_audio(in_path, out_path)

print("✅ Selesai! Semua file asli telah diproses dan disimpan di folder Processed.")
