import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from Scripts.Data_Collector.acoustic_feature_collector import AcousticFeatures

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Collate Function ----------------
def collate_fn(batch):
    features, labels, filenames = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)  # (batch, max_seq_len, feat_dim)
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
        # x shape: (batch, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)
        
        # Layer normalization
        summary = self.layer_norm(context_vector)
        
        return lstm_out, summary

# ---------------- Feature Type ----------------
features = ["MFCC", "GFCC", "LogFBank", "f0"]

# ---------------- Extraction ----------------
def extract_temporal_features(features, hidden_dim=128, num_layers=2, batch_size=32):
    for feat in features:
        print(f"\n🎵 Ekstraksi fitur temporal dengan LSTM untuk {feat}")

        # dataset + dataloader
        train_dataset = AcousticFeatures(model_state="train", feature_type=feat)
        test_dataset  = AcousticFeatures(model_state="test",  feature_type=feat)
        # eval_dataset  = AcousticFeatures(model_state="eval",  feature_type=feat)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        # eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # ---------------- Model ----------------
        sample_feat, _, _ = train_dataset[0]
        input_dim = sample_feat.shape[1]

        model = LSTMFeatureExtractor(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
        ).to(device)
        model.eval()

        # ---------------- Simpan Output ----------------
        save_dir_train = f"Dataset/Temporal_Features/train/{feat}"
        save_dir_test  = f"Dataset/Temporal_Features/test/{feat}"
        # save_dir_eval  = f"Dataset/Temporal_Features/eval/{feat}"
        os.makedirs(save_dir_train, exist_ok=True)
        os.makedirs(save_dir_test,  exist_ok=True)
        # os.makedirs(save_dir_eval, exist_ok=True)

        with torch.no_grad():
            # train
            for X, y, fnames in tqdm(train_loader, desc=f"[{feat}] Saving LSTM feats (train)"):
                X = X.to(device)
                temporal, summary = model(X)
                for i, fname in enumerate(fnames):
                    np.save(os.path.join(save_dir_train, f"{fname}_temporal.npy"), temporal[i].cpu().numpy())
                    np.save(os.path.join(save_dir_train, f"{fname}_summary.npy"),  summary[i].cpu().numpy())

            # test
            for X, y, fnames in tqdm(test_loader, desc=f"[{feat}] Saving LSTM feats (test)"):
                X = X.to(device)
                temporal, summary = model(X)
                for i, fname in enumerate(fnames):
                    np.save(os.path.join(save_dir_test, f"{fname}_temporal.npy"), temporal[i].cpu().numpy())
                    np.save(os.path.join(save_dir_test, f"{fname}_summary.npy"),  summary[i].cpu().numpy())

            # eval
            # for X, y, fnames in tqdm(eval_loader, desc=f"[{feat}] Saving LSTM feats (eval)"):
            #     X = X.to(device)
            #     temporal, summary = model(X)
            #     for i, fname in enumerate(fnames):
            #         np.save(os.path.join(save_dir_eval, f"{fname}_temporal.npy"), temporal[i].cpu().numpy())
            #         np.save(os.path.join(save_dir_eval, f"{fname}_summary.npy"),  summary[i].cpu().numpy())

        print(f"✅ Semua output LSTM (temporal & summary) untuk {feat} sudah disimpan.")

# ---------------- Run ----------------
if __name__ == "__main__":
    extract_temporal_features(features)
