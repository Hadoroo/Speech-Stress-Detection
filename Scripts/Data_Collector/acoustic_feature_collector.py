import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class AcousticFeatures(Dataset):
    def __init__(self, model_state, dataset_name, feature_type="MFCC"):
        """
        model_state: "train" atau "test"
        dataset_name: nama dataset (misal: "RAVDESS", "TESS", "CREMAD")
        feature_type: "MFCC", "GFCC", "LogFBank", atau "F0"
        """
        self.dataset_name = dataset_name
        self.feature_type = feature_type

        # Path CSV dan fitur akustik
        csv_path = f"Dataset/{dataset_name}/CSV/{model_state}_split_stress.csv"
        feature_dir = f"Dataset/{dataset_name}/Acoustic_Features/{model_state}/{feature_type}"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ CSV tidak ditemukan: {csv_path}")
        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"❌ Folder fitur tidak ditemukan: {feature_dir}")

        self.data = pd.read_csv(csv_path)
        self.feature_dir = feature_dir

        # Mapping label (Stress / Non-stress)
        unique_labels = sorted(self.data["stress"].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = os.path.splitext(row["filename"])[0]

        feature_path = os.path.join(
            self.feature_dir,
            f"{filename}_{self.feature_type.lower()}.npy"
        )

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"❌ File fitur tidak ditemukan: {feature_path}")

        features = np.load(feature_path)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.label_map[row["stress"]], dtype=torch.long)

        return features, label, filename
