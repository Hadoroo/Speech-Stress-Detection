import os
from scipy.io import wavfile
import numpy as np

folder_path = "Dataset/CREMAD/Processed"  # sesuaikan path

durations = []

for file in os.listdir(folder_path):
    if file.endswith(".wav"):
        path = os.path.join(folder_path, file)
        sr, data = wavfile.read(path)
        duration = len(data) / sr
        durations.append(duration)

durations = np.array(durations)

print(f"Jumlah file        : {len(durations)}")
print(f"Durasi minimum     : {durations.min():.2f} detik")
print(f"Durasi maksimum    : {durations.max():.2f} detik")
print(f"Rata-rata durasi   : {durations.mean():.2f} detik")
