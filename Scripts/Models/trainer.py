import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cnn import AudioCNN
from Scripts.Data_Loader.acoustic_dataloader import AcousticDataset

# =========================
# CONFIG
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
N_FOLDS = 5

BASE_PATH = "Dataset/CREMAD"
FEATURE_ROOT = os.path.join(BASE_PATH, "Acoustic_Features")
KFOLD_PATH = os.path.join(BASE_PATH, "Split/Chyan/kfold")

CSV_PATH = "Results/Tuning/cnn_results.csv"

# =========================
# LOAD BEST CONFIG PER FEATURE
# =========================
df = pd.read_csv(CSV_PATH)

best_configs = (
    df.sort_values("test_acc", ascending=False)
      .groupby("feature")
      .head(1)
      .reset_index(drop=True)
)

print("\nBEST CONFIG PER FEATURE")
print(best_configs)

# =========================
# LOAD METADATA
# =========================
metadata = pd.read_csv(os.path.join(KFOLD_PATH, "metadata.csv"))

final_results = []

# =========================
# FINAL K-FOLD TRAINING
# =========================
for _, row in tqdm(
    best_configs.iterrows(),
    total=len(best_configs),
    desc="Final KFold Training"
):

    feature = row["feature"]

    # convert string representation back to list
    if "+" in feature:
        feature_list = feature.split("+")
    else:
        feature_list = [feature]

    BEST_CONFIG = {
        "k1": int(row["k1"]),
        "k2": int(row["k2"]),
        "dropout1": float(row["dropout1"]),
        "dropout2": float(row["dropout2"]),
        "batch_size": int(row["batch_size"]),
        "lr": float(row["lr"]),
    }

    print(f"\n▶ Feature: {feature}")
    print("Using config:", BEST_CONFIG)

    dataset = AcousticDataset(
        FEATURE_ROOT,
        metadata,
        feature_type=feature_list
    )

    fold_accuracies = []

    for fold in range(1, N_FOLDS + 1):

        print(f"\nFold {fold}")

        split = np.load(
            os.path.join(KFOLD_PATH, f"fold_{fold}.npz")
        )

        train_idx = split["train_idx"]
        val_idx = split["val_idx"]

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=BEST_CONFIG["batch_size"],
            shuffle=True
        )

        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=BEST_CONFIG["batch_size"],
            shuffle=False
        )

        model = AudioCNN(
            num_classes=metadata["label"].nunique(),
            k1=BEST_CONFIG["k1"],
            k2=BEST_CONFIG["k2"],
            dropout1=BEST_CONFIG["dropout1"],
            dropout2=BEST_CONFIG["dropout2"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=BEST_CONFIG["lr"])
        criterion = nn.CrossEntropyLoss()

        # =========================
        # TRAIN
        # =========================
        for epoch in range(EPOCHS):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                pred = model(x).argmax(dim=1).cpu().numpy()
                y_pred.extend(pred)
                y_true.extend(y.numpy())

        acc = 100 * (np.array(y_true) == np.array(y_pred)).mean()
        fold_accuracies.append(acc)

        print(f"Fold {fold} Accuracy: {acc:.2f}%")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"\n{feature} Final Mean Acc: {mean_acc:.2f}% ± {std_acc:.2f}")

    final_results.append({
        "feature": feature,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        **BEST_CONFIG
    })

# =========================
# SAVE FINAL RESULTS
# =========================
df_final = pd.DataFrame(final_results)
df_final.to_csv("final_kfold_results.csv", index=False)

print("\nFINAL K-FOLD RESULTS SAVED → final_kfold_results.csv")