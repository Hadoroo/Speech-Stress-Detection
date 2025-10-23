import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset

class TemporalFeatures(Dataset):
    def __init__(self, dataset_name, model_state, acoustic_feature_type="MFCC", temporal_feature_type="temporal"):
        """
        dataset_name: nama dataset (misalnya 'RAVDESS' atau 'TESS')
        model_state: 'train' atau 'test'
        acoustic_feature_type: 'MFCC', 'GFCC', atau 'LogFBank'
        temporal_feature_type: nama fitur temporal (misalnya 'temporal', 'delta', dst)
        """
        self.data = pd.read_csv(f"Dataset/{dataset_name}/CSV/{model_state}_split_stress.csv")
        self.feature_dir = f"Dataset/{dataset_name}/Temporal_Features/{model_state}/{acoustic_feature_type}"
        self.acoustic_feature_type = acoustic_feature_type
        self.temporal_feature_type = temporal_feature_type
        
        unique_labels = sorted(self.data["stress"].unique())       
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = os.path.splitext(row["filename"])[0]  # tanpa .wav

        # Path file fitur
        feature_path = os.path.join(
            self.feature_dir,
            f"{filename}_{self.temporal_feature_type}.npy"
        )

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        features = np.load(feature_path)  # shape: (time_steps, feat_dim)

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.label_map[row["stress"]], dtype=torch.long)

        return features, label, filename
