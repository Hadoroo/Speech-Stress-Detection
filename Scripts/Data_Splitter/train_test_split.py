import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

LABEL_MAP = {
    "ANG": "Stressed",
    "DIS": "Stressed",
    "FEA": "Stressed",
    "HAP": "Unstressed",
    "NEU": "Unstressed",
    "SAD": "Stressed",
}

BASE_FOLDER_PATH = "Dataset/CREMAD"
FEATURE_ROOT = os.path.join(BASE_FOLDER_PATH, "Raw")
SPLIT_PATH = os.path.join(BASE_FOLDER_PATH, "Split/Chyan/train_test")
os.makedirs(SPLIT_PATH, exist_ok=True)

print(f"\nMemproses dataset: CREMA-D")

data = []

for file in sorted(os.listdir(FEATURE_ROOT)):
    if not file.endswith(".wav"):
        continue

    # contoh: 1001_DFA_ANG_XX.wav
    parts = file.replace(".wav", "").split("_")
    if len(parts) < 4:
        continue

    actor = parts[0]          # 1001
    emotion = parts[2]        # ANG
    stress = LABEL_MAP.get(emotion)

    if stress is None:
        continue

    data.append({
        "filename": file,
        "actor": actor,
        "emotion": emotion,
        "stress": stress,
        "label": 1 if stress == "Stressed" else 0
    })

df = pd.DataFrame(data)
df["filename"] = df["filename"].str.replace(".wav", ".npy", regex=False)

print(f"Total samples: {len(df)}")
print("Distribusi stress label:")
print(df["stress"].value_counts())
assert len(df) > 0, "ERROR: Data kosong"

# ===================== GROUP TRAIN-TEST SPLIT 80/20 =====================
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)

X_dummy = np.zeros(len(df))
y = df["label"].values
groups = df["actor"].values

train_idx, test_idx = next(splitter.split(X_dummy, y, groups))

np.savez(os.path.join(SPLIT_PATH, "train_test_80_20_group_actor.npz"),
         train_idx=train_idx,
         test_idx=test_idx)

print(f"\nDataset berhasil di-split dengan Group Actor 80:20")
print(f"Train: {len(train_idx)} samples")
print(f"Test : {len(test_idx)} samples")

# Simpan metadata
csv_path = os.path.join(SPLIT_PATH, "metadata.csv")
df.to_csv(csv_path, index=False)
print(f"\nMetadata disimpan pada: {csv_path}")
print("Splitting selesai — siap untuk training!")
