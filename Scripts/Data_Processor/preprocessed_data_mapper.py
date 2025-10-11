import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ---------------- Paths ----------------
dataset_folder = "Dataset/Processed"
os.makedirs("Dataset/CSV", exist_ok=True)

# ---------------- Load all audio files ----------------
files = [f for f in os.listdir(dataset_folder) if f.endswith(".wav")]

# ---------------- Emotion → Stress mapping ----------------
stress_map = {
    "ANG": "High-stress",
    "SAD": "High-stress",
    "FEA": "Low-stress",   
    "DIS": "Low-stress",   
    "NEU": "Non-stress",
    "HAP": "Non-stress"
}

# ---------------- Parse filename metadata ----------------
data = []
for f in files:
    parts = f.split("_")
    if len(parts) < 4:  # skip file dengan format tidak valid
        continue

    actor = parts[0]                  # nomor aktor
    sentence = parts[1]               # kalimat
    emotion = parts[2]                # emosi
    pitch = parts[3].split(".")[0]    # intonasi (HI, LO, MD, XX)
    
    stress = stress_map.get(emotion, "unknown")

    data.append([f, actor, sentence, emotion, pitch, stress])

# Buat DataFrame
df = pd.DataFrame(data, columns=["filename", "actor", "sentence", "emotion", "pitch", "stress"])

# ---------------- Data cleaning ----------------
# 1️⃣ Hapus emosi yang tidak dipetakan
df = df[df["stress"] != "unknown"]

# 2️⃣ Hapus pitch yang tidak diketahui (XX)
df = df[df["pitch"].str.upper() != "XX"]

# 3️⃣ (Opsional) reset index agar rapi
df = df.reset_index(drop=True)

# ---------------- Speaker-independent split ----------------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["actor"]))

df_train = df.iloc[train_idx]
df_test  = df.iloc[test_idx]

# ---------------- Print summary ----------------
print(f"Total data: {len(df)} (setelah filter pitch XX dan unknown stress)\n")

print("Actor train:", df_train["actor"].nunique(), "→", sorted(df_train["actor"].unique()))
print("Actor test:", df_test["actor"].nunique(), "→", sorted(df_test["actor"].unique()))

print("\nTrain distribution:", df_train["stress"].value_counts().to_dict())
print("Test distribution:", df_test["stress"].value_counts().to_dict())

# ---------------- Save CSV ----------------
df_train.to_csv("Dataset/CSV/train_split_stress.csv", index=False)
df_test.to_csv("Dataset/CSV/test_split_stress.csv", index=False)

print("\n✅ Split speaker-independent selesai dan disimpan di folder Dataset/CSV/")
