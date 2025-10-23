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
    features_padded = pad_sequence(features, batch_first=True)
    labels = torch.tensor(labels)
    return features_padded, labels, filenames

# ---------------- LSTM Model ----------------
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        summary = self.layer_norm(context_vector)
        return lstm_out, summary

# ---------------- Feature & Dataset Lists ----------------
features = ["MFCC", "GFCC", "LogFBank", "F0"]
datasets = ["TESS"]

# ---------------- Extraction Function ----------------
def extract_temporal_features(features, datasets, hidden_dim=128, num_layers=2, batch_size=32):
    for dataset_name in datasets:
        print(f"\n📂 Dataset: {dataset_name}")
        for feat in features:
            print(f"🎵 Ekstraksi fitur temporal LSTM untuk: {feat}")

            # Dataset & DataLoader
            train_dataset = AcousticFeatures(model_state="train", dataset_name=dataset_name, feature_type=feat)
            test_dataset  = AcousticFeatures(model_state="test",  dataset_name=dataset_name, feature_type=feat)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            # Model
            sample_feat, _, _ = train_dataset[0]
            input_dim = sample_feat.shape[1]

            model = LSTMFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(device)
            model.eval()

            # Path penyimpanan
            save_dir_train = f"Dataset/{dataset_name}/Temporal_Features/train/{feat}"
            save_dir_test  = f"Dataset/{dataset_name}/Temporal_Features/test/{feat}"
            os.makedirs(save_dir_train, exist_ok=True)
            os.makedirs(save_dir_test, exist_ok=True)

            with torch.no_grad():
                # Train
                for X, y, fnames in tqdm(train_loader, desc=f"[{dataset_name}][{feat}] Saving train"):
                    X = X.to(device)
                    temporal, summary = model(X)
                    for i, fname in enumerate(fnames):
                        np.save(os.path.join(save_dir_train, f"{fname}_temporal.npy"), temporal[i].cpu().numpy())
                        np.save(os.path.join(save_dir_train, f"{fname}_summary.npy"),  summary[i].cpu().numpy())

                # Test
                for X, y, fnames in tqdm(test_loader, desc=f"[{dataset_name}][{feat}] Saving test"):
                    X = X.to(device)
                    temporal, summary = model(X)
                    for i, fname in enumerate(fnames):
                        np.save(os.path.join(save_dir_test, f"{fname}_temporal.npy"), temporal[i].cpu().numpy())
                        np.save(os.path.join(save_dir_test, f"{fname}_summary.npy"),  summary[i].cpu().numpy())

            print(f"✅ Selesai ekstraksi fitur temporal & summary untuk {feat} dari {dataset_name}.\n")

# ---------------- Run ----------------
if __name__ == "__main__":
    extract_temporal_features(features, datasets)
