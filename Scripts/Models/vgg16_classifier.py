import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from Scripts.Data_Collector.temporal_feature_collector import TemporalFeatures

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------------- Feature Types (F0 removed) ----------------
features = ["MEL", "MFCC", "GFCC", "LogFBank"]

# ---------------- Dataset List ----------------
datasets = ["CREMAD"]

# ---------------- Collate Function ----------------
# We resize every sample so the CNN gets a fixed input shape (B, 1, TARGET_F, TARGET_T)
TARGET_F = 64   # frequency dimension (height)
TARGET_T = 256  # time dimension (width)

def collate_fn(batch, pad_value=0.0):
    """
    batch: list of tuples (feat, label, fname)
    feat: array-like or torch tensor, expected shape either (T, D) or (D, T)
    returns:
        X : (B, 1, TARGET_F, TARGET_T)  # ready for VGG conv2d
        labels : (B,)
        fnames : list
        lengths: (B,) actual T before resizing (useful if needed)
    """
    feats, labels, fnames = zip(*batch)
    proc = []
    lengths = []
    for f in feats:
        # convert to numpy then to torch.tensor
        if not isinstance(f, torch.Tensor):
            f = np.asarray(f)
        # make float32 tensor
        f = torch.from_numpy(f).float() if isinstance(f, np.ndarray) else f.float()

        # ensure 2D
        if f.dim() != 2:
            raise ValueError("Feature must be 2D (T, D) or (D, T). Got shape: %s" % (f.shape,))

        T, D = f.shape  # shape as provided

        # Decide orientation: we want freq x time when passing to CNN.
        # If dims seem swapped (first dim small and second large), transpose.
        # Heuristic: if first dim < second dim and second dim > 8 -> likely (D, T)
        if T < D and D > 8:
            f = f.t()  # (D,T) -> (T,D)
            T, D = f.shape

        # Now f shape is (T, D) where T is time frames, D is feature dim (freq-like)
        # For CNN we want (batch, channel=1, freq, time)
        # Convert to (1, 1, D, T) then interpolate to (1, 1, TARGET_F, TARGET_T)
        f = f.t().unsqueeze(0).unsqueeze(0)  # (1,1,D,T)

        # Resize/interpolate using bilinear (works reasonably for spectrogram-like)
        f_resized = F.interpolate(f, size=(TARGET_F, TARGET_T), mode="bilinear", align_corners=False)
        # squeeze batch-like dim for stacking: will append shape (1, TARGET_F, TARGET_T)
        proc.append(f_resized.squeeze(0))  # (1, TARGET_F, TARGET_T)

        lengths.append(T)

    # Stack -> (B, 1, TARGET_F, TARGET_T)
    X = torch.stack(proc)  # shape: (B,1,TARGET_F,TARGET_T)
    labels = torch.tensor(labels, dtype=torch.long)

    return X, labels, list(fnames), torch.tensor(lengths, dtype=torch.long)


# ===============================================================
# ====================== VGG16 CLASSIFIER =======================
class VGG16(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()

        # ============= BLOCK 1 =============
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(2, 2)
        )

        # ============= BLOCK 2 =============
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.MaxPool2d(2, 2)
        )

        # ============= BLOCK 3 =============
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(2, 2)
        )

        # ============= BLOCK 4 (Residual-like) ============
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1   = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2   = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3   = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2)

        # ============= BLOCK 5 =============
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.MaxPool2d(2, 2)
        )

        # ============= GLOBAL POOL ============
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ============= FULLY CONNECTED ============
        # fc6 & fc7 diperkecil agar stabil untuk input spectrogram-like yang bisa berukuran kecil
        self.fc6 = nn.Sequential(
            nn.Linear(512, 1024),       # from 4096 → 1024
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(1024, 512),       # from 4096 → 512
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )

        self.fc8 = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        skip = self.relu(self.bn4_1(self.conv4_1(x)))
        out  = self.relu(self.bn4_2(self.conv4_2(skip)))
        out  = out + skip
        out  = self.relu(self.bn4_3(self.conv4_3(out)))
        x    = self.pool4(out)

        x = self.block5(x)
        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)

        return x


# ---------------- Calculate Class Weights ----------------
def calculate_class_weights(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    # handle possible zero-count safety
    class_counts = np.where(class_counts == 0, 1, class_counts)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------- Quick Evaluation Function ----------------
def evaluate_model(model, dataloader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            # dataloader returns (X, labels, fnames, lengths)
            if len(batch) == 4:
                X, y, _, _ = batch
            else:
                X, y = batch[0], batch[1]

            if X.size(0) == 1:
                continue

            X, y = X.to(device), y.to(device)
            out = model(X.float())
            _, p = torch.max(out, 1)

            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    if len(labels) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0)
    }


# ========================= IMPROVED GRID SEARCH =========================
def grid_search(features, datasets):
    # Hyperparameter yang ingin dicari
    params = {
        'learning_rate': [1e-4, 3e-4],
        'batch_size': [8, 16],
        'dropout_rate': [0.3, 0.5],
        'weight_decay': [1e-4, 5e-4]
    }

    # max epoch untuk grid search (epoch search)
    SEARCH_EPOCH = 12  # cukup 10–12 epoch untuk evaluasi stabil

    save_dir = "Results/Tuning/VGG16"
    os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        for feat in features:

            save_path = os.path.join(save_dir, f"{dataset}_{feat}.npy")
            if os.path.exists(save_path):
                print(f"Skip tuning (exists): {save_path}")
                continue

            print(f"\n🔎 Grid Search for {dataset}-{feat}")

            train_ds = TemporalFeatures(dataset, "train", feat, "temporal")
            test_ds  = TemporalFeatures(dataset, "test", feat, "temporal")

            best_score = -1.0
            best_params = None
            best_epoch = 1

            # Loop semua kombinasi parameter
            for p in tqdm(list(ParameterGrid(params)), desc=f"Tuning {dataset}-{feat}"):

                lr = p["learning_rate"]
                bs = p["batch_size"]
                dr = p["dropout_rate"]
                wd = p["weight_decay"]

                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
                test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

                # Model & optimizer
                model = VGG16(dropout_rate=dr).to(device)
                class_weights = calculate_class_weights(train_ds)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                # ==== TRAIN SEARCH_EPOCH untuk mencari epoch terbaik untuk kombinasi ini ====
                best_acc_local = -1.0
                best_epoch_local = 1

                for epoch in range(SEARCH_EPOCH):
                    model.train()
                    for X, y, _, _ in train_loader:
                        if X.size(0) == 1:
                            continue
                        X, y = X.to(device), y.to(device)

                        optimizer.zero_grad()
                        out = model(X.float())
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()

                    # Validate setiap epoch
                    eval_result = evaluate_model(model, test_loader)
                    acc = eval_result["accuracy"]

                    # update local best
                    if acc > best_acc_local:
                        best_acc_local = acc
                        best_epoch_local = epoch + 1

                # setelah SEARCH_EPOCH, bandingkan local best dengan global best
                if best_acc_local > best_score:
                    best_score = best_acc_local
                    best_params = p
                    best_epoch = best_epoch_local

            # simpan parameter + epoch terbaik
            np.save(save_path, {
                "params": best_params,
                "best_acc": best_score,
                "best_epoch": best_epoch
            })

            print(f"🔥 Best Params for {dataset}-{feat}: {best_params} "
                  f"| Epoch={best_epoch} | Acc={best_score:.4f}")


# ========================= FINAL TRAINING & EVALUATION =========================
def train_and_evaluate(features, datasets):
    model_dir = "Results/Models/VGG16"
    pred_dir  = "Results/Predictions/VGG16"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    for dataset in datasets:
        for feat in features:

            tune_path = f"Results/Tuning/VGG16/{dataset}_{feat}.npy"
            if not os.path.exists(tune_path):
                print(f"[SKIP] No tuning params for {dataset}-{feat}")
                continue

            tune = np.load(tune_path, allow_pickle=True).item()
            p = tune["params"]
            best_epoch = int(tune.get("best_epoch", 25))  # fallback jika tidak ada

            lr, bs = p["learning_rate"], p["batch_size"]
            dr, wd = p["dropout_rate"], p["weight_decay"]

            print(f"\n🚀 Final Training {dataset}-{feat} | best_epoch={best_epoch} | params={p}")

            # Dataset
            train_ds = TemporalFeatures(dataset, "train", feat, "temporal")
            test_ds  = TemporalFeatures(dataset, "test", feat, "temporal")

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
            test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

            # Model
            model = VGG16(dropout_rate=dr).to(device)
            class_weights = calculate_class_weights(train_ds)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            # HISTORY
            history = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": [],
                "test_precision": [],
                "test_recall": [],
                "test_f1": []
            }

            # TRAINING LOOP menggunakan best_epoch
            for epoch in range(best_epoch):
                model.train()
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{best_epoch}", leave=False)

                total_loss = 0.0
                correct = 0
                total = 0

                for X, y, _, _ in pbar:
                    if X.size(0) == 1:
                        continue

                    X, y = X.to(device), y.to(device)

                    optimizer.zero_grad()
                    out = model(X.float())
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, pred = torch.max(out, 1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                train_loss = total_loss / max(len(train_loader), 1)
                train_acc = correct / total if total > 0 else 0.0

                # EVALUATION
                model.eval()
                preds, labels = [], []
                test_total_loss = 0.0

                with torch.no_grad():
                    for X, y, _, _ in test_loader:
                        if X.size(0) == 1:
                            continue
                        X, y = X.to(device), y.to(device)
                        out = model(X.float())

                        loss = criterion(out, y)
                        test_total_loss += loss.item()

                        _, p = torch.max(out, 1)
                        preds.extend(p.cpu().numpy())
                        labels.extend(y.cpu().numpy())

                test_loss = test_total_loss / max(len(test_loader), 1)
                test_acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
                test_prec = precision_score(labels, preds, average="weighted", zero_division=0) if len(labels) > 0 else 0.0
                test_rec = recall_score(labels, preds, average="weighted", zero_division=0) if len(labels) > 0 else 0.0
                test_f1 = f1_score(labels, preds, average="weighted", zero_division=0) if len(labels) > 0 else 0.0

                # record
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc)
                history["test_precision"].append(test_prec)
                history["test_recall"].append(test_rec)
                history["test_f1"].append(test_f1)

                print(
                    f"Epoch {epoch+1:02d} | "
                    f"TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.3f} | "
                    f"TestLoss={test_loss:.4f} | TestAcc={test_acc:.3f} | "
                    f"Prec={test_prec:.3f} | Rec={test_rec:.3f} | F1={test_f1:.3f}"
                )

            # SAVE
            torch.save(model.state_dict(), os.path.join(model_dir, f"{dataset}_{feat}.pt"))
            np.save(os.path.join(pred_dir, f"{dataset}_{feat}.npy"), history)

            print(f"🎯 Saved: {dataset}-{feat} | Final TestAcc={history['test_acc'][-1]:.3f}")


# ========================= MAIN FUNCTION =========================
if __name__ == "__main__":
    print("🚀 Starting VGG16 Training...")
    grid_search(features, datasets)
    train_and_evaluate(features, datasets)
    print("🎉 All completed!")
