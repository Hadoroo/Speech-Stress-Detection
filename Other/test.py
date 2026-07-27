

import numpy as np
import pandas as pd

# =========================
# 1. LOAD FILE NPY
# =========================
file_path = "Dataset/CREMAD/Acoustic_Features/MFCC/1001_DFA_ANG_XX.npy"  # ganti dengan file kamu
data = np.load(file_path)

print("=== INFO DATA ===")
print("Shape data:", data.shape)
print("Tipe data:", type(data))
print()

# =========================
# 2. AMBIL 1 SAMPLE
# =========================
# Kalau banyak data (misal: (N, ...))
if len(data.shape) >= 2:
    sample = data[0]
else:
    sample = data

print("=== SAMPLE AWAL ===")
print("Shape sample:", sample.shape)
print()

# =========================
# 3. HANDLE DIMENSI
# =========================

# Case 3D (misalnya MFCC dengan channel)
if len(sample.shape) == 3:
    print("Detected 3D (channel, height, width) → ambil channel pertama")
    sample = sample[0]

# Case 1D (flatten)
elif len(sample.shape) == 1:
    print("Detected 1D (flatten) → mencoba reshape otomatis")

    length = sample.shape[0]

    # coba bentuk mendekati persegi
    h = int(np.sqrt(length))
    while length % h != 0:
        h -= 1
    w = length // h

    print(f"Reshape ke ({h}, {w})")
    sample = sample.reshape(h, w)

print("=== SAMPLE SETELAH DIPROSES ===")
print("Shape:", sample.shape)
print()

# =========================
# 4. OPSIONAL: KECILKAN UKURAN (UNTUK EXCEL)
# =========================
max_size = 5  # ubah kalau mau

if sample.shape[0] > max_size or sample.shape[1] > max_size:
    print(f"Memotong sample jadi {max_size}x{max_size} untuk kemudahan Excel")
    sample = sample[:max_size, :max_size]

print("=== SAMPLE FINAL ===")
print(sample)
print()

# =========================
# 5. CEK NILAI
# =========================
print("Min:", np.min(sample))
print("Max:", np.max(sample))
print("Jumlah non-zero:", np.count_nonzero(sample))
print()

