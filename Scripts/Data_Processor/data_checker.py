import os
import librosa
import numpy as np

SR = 16000
N_FFT = 2048
HOP = 512

INPUT_DIR = "Dataset/CREMAD/Raw"

frame_lengths = []
durations = []

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if not file.endswith(".wav"):
            continue

        path = os.path.join(root, file)

        # load audio
        y, sr = librosa.load(path, sr=SR)

        # durasi audio
        duration = librosa.get_duration(y=y, sr=sr)
        durations.append(duration)

        # hitung jumlah frame
        frames = librosa.util.frame(
            y,
            frame_length=N_FFT,
            hop_length=HOP
        )

        frame_lengths.append(frames.shape[1])

# statistik
print("===== DURASI AUDIO =====")
print(f"Rata-rata : {np.mean(durations):.2f} detik")
print(f"Minimum   : {np.min(durations):.2f} detik")
print(f"Maksimum  : {np.max(durations):.2f} detik")

print("\n===== JUMLAH FRAME =====")
print(f"Rata-rata : {np.mean(frame_lengths):.2f}")
print(f"Median    : {np.median(frame_lengths):.2f}")
print(f"Minimum   : {np.min(frame_lengths)}")
print(f"Maksimum  : {np.max(frame_lengths)}")