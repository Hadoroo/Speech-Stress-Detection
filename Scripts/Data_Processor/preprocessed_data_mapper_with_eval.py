import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split

# Folder dataset
dataset_folder = "Dataset/Processed"

# Ambil semua file audio
files = [f for f in os.listdir(dataset_folder) if f.endswith(".wav")]

# Mapping emosi ke stress level
stress_map = {
    "ANG": "High-stress",
    "SAD": "High-stress",
    "FEA": "Low-stress",   
    "DIS": "Low-stress",   
    "NEU": "Non-stress",
    "HAP": "Non-stress"
}

# Ekstrak informasi dari nama file
data = []
for f in files:
    parts = f.split("_")
    actor = parts[0]      # nomor aktor
    sentence = parts[1]   # kalimat (dfa, dfd, dll.)
    emotion = parts[2]    # emosi asli
    pitch = parts[3].split(".")[0]  # intonasi (XX)
    stress = stress_map.get(emotion, "unknown")  # map ke stress level
    data.append([f, actor, sentence, emotion, pitch, stress])

df = pd.DataFrame(data, columns=["filename", "actor", "sentence", "emotion", "pitch", "stress"])

# Buang data dengan label unknown (emosi yg tidak dipetakan)
df = df[df["stress"] != "unknown"]

# Stratified split berdasarkan stress
train_files, temp_files, train_labels, temp_labels = train_test_split(
    df["filename"], df["stress"], 
    test_size=0.2, 
    stratify=df["stress"], 
    random_state=42
)

test_files, eval_files = train_test_split(
    temp_files, 
    test_size=0.5, 
    stratify=temp_labels, 
    random_state=42
)

# Buat DataFrame hasil split
df_train = df[df["filename"].isin(train_files)]
df_test  = df[df["filename"].isin(temp_files)]
df_eval  = df[df["filename"].isin(eval_files)]

print("Train:", df_train["stress"].value_counts().to_dict())
print("Test:", df_test["stress"].value_counts().to_dict())
print("Eval:", df_eval["stress"].value_counts().to_dict())

# Simpan CSV
df_train.to_csv("Dataset/CSV/train_split_stress.csv", index=False)
df_test.to_csv("Dataset/CSV/test_split_stress.csv", index=False)
df_eval.to_csv("Dataset/CSV/eval_split_stress.csv", index=False)
