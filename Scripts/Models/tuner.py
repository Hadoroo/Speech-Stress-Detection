import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from cnn import AudioCNN
from Scripts.Data_Loader.acoustic_dataloader import AcousticDataset

# =========================
# CONFIG
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20

BASE_PATH = "Dataset/CREMAD"
FEATURE_ROOT = os.path.join(BASE_PATH, "Acoustic_Features")
SPLIT_PATH = os.path.join(BASE_PATH, "Split/Chyan/train_test")

# EARLY FUSION COMBINATIONS
FEATURES = [
    ["MFCC"],
    ["GFCC"],
    ["LPC"],
    ["MFCC", "GFCC"],
    ["MFCC", "LPC"],
    ["GFCC", "LPC"],
    ["MFCC", "GFCC", "LPC"]
]

param_grid = {
    "k1": [32, 64],
    "k2": [64, 128],
    "dropout1": [0.2, 0.3],
    "dropout2": [0.3, 0.4],
    "batch_size": [16, 32],
    "lr": [1e-3, 1e-4],
}

metadata = pd.read_csv(os.path.join(SPLIT_PATH, "metadata.csv"))

# Load train-test index
split = np.load(os.path.join(SPLIT_PATH, "train_test_80_20_group_actor.npz"))

train_idx = split["train_idx"]
test_idx = split["test_idx"]


results = []

# =========================
# GRID SEARCH
# =========================
for feature_list in tqdm(FEATURES, desc="Feature Grid Search"):

    feature_name = "+".join(feature_list)
    print(f"\nGRID SEARCH FEATURE: {feature_name}")

    dataset = AcousticDataset(
        FEATURE_ROOT,
        metadata,
        feature_type=feature_list
    )

    for values in product(*param_grid.values()):
        cfg = dict(zip(param_grid.keys(), values))

        print(f"\nConfig: {cfg}")

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=cfg["batch_size"],
            shuffle=True
        )

        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=cfg["batch_size"],
            shuffle=False
        )

        model = AudioCNN(
            num_classes=metadata["label"].nunique(),
            k1=cfg["k1"],
            k2=cfg["k2"],
            dropout1=cfg["dropout1"],
            dropout2=cfg["dropout2"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
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
        # TEST
        # =========================
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        test_acc = 100 * correct / total

        results.append({
            "feature": feature_name,
            **cfg,
            "test_acc": test_acc
        })

        print(f"Test Acc: {test_acc:.2f}%")

# =========================
# SAVE RESULT
# =========================
os.makedirs("Results/Tuning", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv("Results/Tuning/cnn_results.csv", index=False)

print("\nTOP 5 CONFIGURATIONS")
print(df.sort_values("test_acc", ascending=False).head(5))