import torch
from torch.utils.data import Dataset
import os
import numpy as np

class AcousticDataset(Dataset):
    def __init__(self, feature_root, metadata_df, feature_type):
        """
        feature_type:
            - "MFCC"
            - ["MFCC", "GFCC"]
            - ["MFCC","GFCC","LPC"]
        """
        self.feature_root = feature_root
        self.metadata = metadata_df.reset_index(drop=True)

        # pastikan selalu list untuk fleksibilitas
        if isinstance(feature_type, str):
            self.feature_type = [feature_type]
        else:
            self.feature_type = feature_type

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        filename = row["filename"]
        label = row["label"]

        feature_list = []

        for feat in self.feature_type:
            path = os.path.join(self.feature_root, feat, filename)
            x = np.load(path)

            # pastikan 2D → (F, T)
            if x.ndim == 1:
                x = x[:, np.newaxis]

            feature_list.append(x)

        # =========================
        # EARLY FUSION
        # =========================
        # gabung di axis frequency
        x = np.concatenate(feature_list, axis=0)

        # pastikan format CNN (C, F, T)
        x = x[np.newaxis, :, :]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )
