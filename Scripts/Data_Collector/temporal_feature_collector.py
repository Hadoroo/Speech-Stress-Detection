import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

class TemporalFeatures(Dataset):
    def __init__(self, model_state, acoustic_feature_type="MFCC", temporal_feature_type="temporal"):
        """
        csv_file: path ke CSV (train/test/eval)
        feature_dir: folder hasil ekstraksi fitur
        feature_type: "MFCC", "GFCC", atau "LogFBank"
        """
        self.data = pd.read_csv(f"Dataset/CSV/{model_state}_split_stress.csv")
        self.feature_dir = f"Dataset/Temporal_Features/{model_state}"
        self.acoustic_feature_type = acoustic_feature_type
        self.temporal_feature_type = temporal_feature_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = os.path.splitext(row["filename"])[0]  # tanpa .wav

        # load fitur npy sesuai jenis
        feature_path = os.path.join(
            self.feature_dir,
            self.acoustic_feature_type,
            f"{filename}_{self.temporal_feature_type}.npy"
        )
        features = np.load(feature_path)  # shape: (time_steps, feat_dim)

        features = torch.tensor(features, dtype=torch.float32)
        label_map = {"High-stress": 0, "Low-stress": 1, "Non-stress": 2}
        label = torch.tensor(label_map[row["stress"]], dtype=torch.long)  # sesuaikan kolom label di CSV

        return features, label, filename