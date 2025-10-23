import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ---------------- Dataset list ----------------
dataset_list = ['CREMAD', 'RAVDESS', 'TESS']
base_folder = "Dataset"

# ---------------- Mapping rules ----------------
# CREMAD emotion → stress mapping
stress_map_cremad = {
    "ANG": "Stress",
    "SAD": "Stress",
    "DIS": "Stress",
    "FEA": "Stress",
    "HAP": "Non-stress",
    "PLE": "Non-stress",
    "SUR": "Non-stress",
    "NEU": "Non-stress"
}

# RAVDESS mapping
ravdess_emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
ravdess_nonstress = ["neutral", "calm", "happy"]

# TESS mapping
tess_nonstress = ["neutral", "happy", "pleasant_surprise"]

# ---------------- Loop per dataset ----------------
for dataset in dataset_list:
    print(f"\n📂 Memproses dataset: {dataset}")
    processed_path = os.path.join(base_folder, dataset, "Processed")
    csv_path = os.path.join(base_folder, dataset, "CSV")
    os.makedirs(csv_path, exist_ok=True)

    if not os.path.exists(processed_path):
        print(f"⚠️  Folder tidak ditemukan: {processed_path}")
        continue

    files = [f for f in os.listdir(processed_path) if f.lower().endswith(".wav")]
    data = []

    for f in files:
        if dataset == "CREMAD":
            parts = f.replace(".wav", "").split("_")
            if len(parts) != 4:
                continue
            actor, sentence, emotion, intensity = parts
            stress = stress_map_cremad.get(emotion, "unknown")
            if stress != "unknown":
                data.append([f, actor, emotion, stress])

        elif dataset == "RAVDESS":
            parts = f.replace(".wav", "").split("-")
            if len(parts) < 7:
                continue
            emotion_code = parts[2]
            actor = parts[-1]
            emotion = ravdess_emotions.get(emotion_code, "unknown")
            if emotion != "unknown":
                stress = "Non-stress" if emotion in ravdess_nonstress else "Stress"
                data.append([f, actor, emotion, stress])

        elif dataset == "TESS":
            # Contoh nama: OAF_back_fear.wav → actor=OAF, emotion=fear
            name = f.replace(".wav", "")
            parts = name.split("_")
            if len(parts) < 3:
                continue
            actor = parts[0]               # OAF / YAF
            emotion = parts[-1].lower()    # ambil bagian terakhir (emosi sebenarnya)
            stress = "Non-stress" if emotion in tess_nonstress else "Stress"
            data.append([f, actor, emotion, stress])

    # Buat DataFrame
    df = pd.DataFrame(data, columns=["filename", "actor", "emotion", "stress"])

    if len(df) == 0:
        print(f"⚠️  Tidak ada file valid di {dataset}")
        continue

    # ---------------- Speaker-independent split ----------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["actor"]))

    df_train = df.iloc[train_idx]
    df_test  = df.iloc[test_idx]

    # ---------------- Print summary ----------------
    print(f"Total data: {len(df)}")
    print("Actor train:", df_train["actor"].nunique(), "→", sorted(df_train["actor"].unique()))
    print("Actor test:", df_test["actor"].nunique(), "→", sorted(df_test["actor"].unique()))
    print("Distribusi train:", df_train["stress"].value_counts().to_dict())
    print("Distribusi test :", df_test["stress"].value_counts().to_dict())

    # ---------------- Save CSV ----------------
    train_file = os.path.join(csv_path, "train_split_stress.csv")
    test_file  = os.path.join(csv_path, "test_split_stress.csv")

    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)

    print(f"✅ Split selesai → disimpan di {csv_path}")

print("\n🎉 Semua dataset telah selesai diproses!")
