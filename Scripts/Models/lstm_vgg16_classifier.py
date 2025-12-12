# train_hybrid.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ganti import ini sesuai lokasi dataset class kamu
from Scripts.Data_Collector.acoustic_feature_collector import AcousticFeatures

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

# ---------------- collate (pad sequence) ----------------
def collate_fn(batch):
    # batch: list of tuples (feat (T,F) tensor, label, filename)
    feats, labels, fnames = zip(*batch)
    # pad sequences (B, T_max, F)
    padded = pad_sequence(feats, batch_first=True)  # zeros padded at end
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels, fnames

# ---------------- LSTM extractor (trainable) ----------------
class LSTMTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, proj_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # project 2*hidden_dim -> proj_dim (smaller channel count for CNN)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x, lengths=None):
        # x: (B, T, F)
        # Optionally, you can pack_padded_sequence if you have lengths
        out, _ = self.lstm(x)           # (B, T, 2H)
        proj = self.proj(out)           # (B, T, proj_dim)
        # transpose for CNN: (B, proj_dim, T)
        proj = proj.transpose(1, 2)
        # make channel dim: (B, 1, proj_dim, T)
        proj = proj.unsqueeze(1)
        return proj  # (B, 1, D, T)

# ---------------- Modified VGG16 (same as kamu gunakan) ----------------
class VGG16Classifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc6 = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # x: (B, C=1, D, T) — D likely small (proj_dim)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        skip = self.relu(self.conv4_1(x))
        out = self.relu(self.conv4_2(skip))
        out = out + skip
        out = self.relu(self.conv4_3(out))
        x = self.pool4(out)
        x = self.block5(x)
        x = self.global_pool(x)   # (B, 512, 1, 1)
        x = x.view(x.size(0), -1) # (B, 512)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x

# ---------------- Hybrid model combining LSTM + VGG ----------------
class HybridLSTM_VGG(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden=128, lstm_layers=2, proj_dim=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.temporal = LSTMTemporal(
            input_dim=lstm_input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=0.2,
            proj_dim=proj_dim
        )
        # VGG expects in_channels = 1 (we feed proj as single-channel map)
        self.vgg = VGG16Classifier(in_channels=1, num_classes=num_classes, dropout_rate=dropout)

    def forward(self, x):
        # x: (B, T, F)
        x = self.temporal(x)    # (B, 1, D, T)
        out = self.vgg(x)
        return out

# ---------------- utilities ----------------
def calculate_class_weights(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    counts = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

def evaluate_model(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y, _ in dataloader:
            if X.size(0) == 1: continue
            X, y = X.to(device), y.to(device)
            out = model(X.float())
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())
    if len(labels) == 0:
        return {"accuracy":0, "precision":0, "recall":0, "f1":0}
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0)
    }

# ---------------- training routine ----------------
def train_end2end(dataset_name="CREMAD", feature="LogFBank",
                  lstm_hidden=128, proj_dim=64, batch_size=16, lr=1e-4, epochs=30):
    print(f"Training end-to-end: {dataset_name} - {feature}")
    train_ds = AcousticFeatures(model_state="train", dataset_name=dataset_name, feature_type=feature)
    test_ds  = AcousticFeatures(model_state="test",  dataset_name=dataset_name, feature_type=feature)

    # sample to get input dim
    sample_feat, _, _ = train_ds[0]  # tensor (T, F)
    input_dim = sample_feat.shape[1]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = HybridLSTM_VGG(lstm_input_dim=input_dim,
                           lstm_hidden=lstm_hidden,
                           lstm_layers=2,
                           proj_dim=proj_dim,
                           num_classes=2,
                           dropout=0.5).to(device)

    class_weights = calculate_class_weights(train_ds)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3
                )


    history = {"train_loss":[], "train_acc":[], "test_acc":[], "test_prec":[], "test_rec":[], "test_f1":[]}

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        total_loss = 0.0
        correct = 0
        total = 0
        for X, y, _ in pbar:
            if X.size(0) == 1: continue
            # ---- optional normalization per-file ----
            # normalize per sample (mean, std along time axis)
            # X = (B, T, F)
            X = X.float()
            mean = X.mean(dim=(1,2), keepdim=True)
            std = X.std(dim=(1,2), keepdim=True) + 1e-8
            X = (X - mean) / std

            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(len(train_loader),1)
        train_acc = correct / total if total>0 else 0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # eval
        metrics = evaluate_model(model, test_loader)
        history["test_acc"].append(metrics["accuracy"])
        history["test_prec"].append(metrics["precision"])
        history["test_rec"].append(metrics["recall"])
        history["test_f1"].append(metrics["f1"])

        scheduler.step(metrics["accuracy"])

        print(f"Epoch {epoch+1}/{epochs} | TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.3f} | TestAcc={metrics['accuracy']:.3f} | F1={metrics['f1']:.3f}")

    # save
    os.makedirs("Results/Models/Hybrid", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("Results/Models/Hybrid", f"{dataset_name}_{feature}_hybrid.pt"))
    np.save(os.path.join("Results/Models/Hybrid", f"{dataset_name}_{feature}_history.npy"), history)
    print("Saved model & history.")
    return history

# ---------------- main ----------------
if __name__ == "__main__":
    # example: run for MFCC, GFCC, LogFBank one by one or change as needed
    feats = ["MEL", "MFCC", "GFCC", "LogFBank"]
    for feat in feats:
        train_end2end(dataset_name="CREMAD", feature=feat,
                      lstm_hidden=128, proj_dim=64, batch_size=16, lr=2e-4, epochs=30)
