import soundfile as sf
import librosa
import numpy as np

file_path = "Dataset/Processed/1001_DFA_ANG_XX.wav"

# --- Cek metadata (bit depth, sample rate, channel)
info = sf.info(file_path)
print(info)

# --- Cek sample rate & channel
y, sr = librosa.load(file_path, sr=None, mono=False)
print("Sample Rate:", sr)
print("Channels:", y.shape[0] if y.ndim > 1 else 1)

# --- Cek normalisasi (dBFS)
peak = np.max(np.abs(y))
dbfs = 20 * np.log10(peak)
print("Peak amplitude (dBFS):", dbfs)
