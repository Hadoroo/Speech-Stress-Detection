import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np

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
SPLIT_PATH = os.path.join(BASE_FOLDER_PATH, "Split/Chyan/kfold")
os.makedirs(SPLIT_PATH, exist_ok=True)

print(f"\nMemproses dataset: CREMA-D")

data = []

for file in sorted(os.listdir(FEATURE_ROOT)):
    if not file.endswith(".wav"):
        continue

    # 1001_DFA_ANG_XX.npy
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
print("Distribusi label:")
print(df["stress"].value_counts())

assert len(df) > 0, "ERROR: Data kosong"


gkf = GroupKFold(n_splits=5)

X_dummy = np.zeros(len(df))
y = df["label"].values
groups = df["actor"].values 

for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X_dummy, y, groups), start=1):

    np.savez(
        os.path.join(SPLIT_PATH, f"fold_{fold}.npz"),
        train_idx=train_idx,
        val_idx=val_idx
    )

    print(f"Fold {fold} disimpan "
          f"(train={len(train_idx)}, val={len(val_idx)})")

csv_path = os.path.join(SPLIT_PATH, "metadata.csv")
df.to_csv(csv_path, index=False)

print(f"\nMetadata disimpan: {csv_path}")
print("KFold split selesai dan siap digunakan")