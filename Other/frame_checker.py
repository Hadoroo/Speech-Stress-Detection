import os
import librosa
import numpy as np

INPUT_DIR = "Dataset/CREMAD/Raw"

durations = []

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            y, sr = librosa.load(path, sr=None)
            duration = len(y) / sr
            durations.append(duration)

durations = np.array(durations)

print("Jumlah file:", len(durations))
print("Mean durasi:", np.mean(durations))
print("Median durasi:", np.median(durations))
print("Max durasi:", np.max(durations))
print("Min durasi:", np.min(durations))