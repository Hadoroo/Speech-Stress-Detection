import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from Scripts.Data_Collector.acoustic_feature_collector import AcousticFeatures

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Collate Function ----------------
def collate_fn(batch):
    features, labels, filenames = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)  # (B, T, F)
    labels = torch.tensor(labels)
    return features_padded, labels, filenames

# ===============================================================
# =============== FINAL & OPTIMAL LSTM FEATURE ==================
# ===============================================================
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, proj_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # reduce 2*hidden_dim → proj_dim = 64
        self.proj = nn.Linear(hidden_dim * 2, proj_dim)

        # layernorm for stability
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)

        # projection → (B, T, D)
        proj = self.proj(lstm_out)
        proj = self.norm(proj)

        # transpose → CNN ready: (B, D, T)
        temporal = proj.transpose(1, 2)

        return temporal


# ---------------- Feature & Dataset Lists ----------------
features = ["MFCC", "GFCC", "LogFBank", "MEL"]
datasets = ["CREMAD"]

# ===============================================================
# ================= Extraction Pipeline ==========================
# ===============================================================
def extract_temporal_features(features, datasets, hidden_dim=128, num_layers=2, batch_size=32):
    for dataset_name in datasets:
        print(f"\n📂 Dataset: {dataset_name}")

        for feat in features:
            print(f"🎵 Ekstraksi fitur temporal dengan LSTM untuk: {feat}")

            # Dataset
            train_dataset = AcousticFeatures(model_state="train", dataset_name=dataset_name, feature_type=feat)
            test_dataset  = AcousticFeatures(model_state="test",  dataset_name=dataset_name, feature_type=feat)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            # Sample size
            sample_feat, _, _ = train_dataset[0]
            input_dim = sample_feat.shape[1]

            # Create model
            model = LSTMFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.3
            ).to(device)
            model.eval()

            # Save directories
            save_dir_train = f"Dataset/{dataset_name}/Temporal_Features/train/{feat}"
            save_dir_test  = f"Dataset/{dataset_name}/Temporal_Features/test/{feat}"
            os.makedirs(save_dir_train, exist_ok=True)
            os.makedirs(save_dir_test, exist_ok=True)

            # ------------------ TRAIN DATA ------------------
            with torch.no_grad():
                for X, y, fnames in tqdm(train_loader, desc=f"[{dataset_name}][{feat}] Saving train"):
                    X = X.to(device)

                    temporal = model(X)

                    for i, fname in enumerate(fnames):
                        np.save(os.path.join(save_dir_train, f"{fname}_temporal.npy"),
                                temporal[i].cpu().numpy())


            # ------------------ TEST DATA ------------------
            with torch.no_grad():
                for X, y, fnames in tqdm(test_loader, desc=f"[{dataset_name}][{feat}] Saving test"):
                    X = X.to(device)

                    temporal = model(X)

                    for i, fname in enumerate(fnames):
                        np.save(os.path.join(save_dir_test, f"{fname}_temporal.npy"),
                                temporal[i].cpu().numpy())

            print(f"✅ Selesai ekstraksi LSTM temporal & summary untuk fitur {feat}.\n")

# ---------------- Run ----------------
if __name__ == "__main__":
    extract_temporal_features(features, datasets)
