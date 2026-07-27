import os
import numpy as np
import torch

from torch.utils.data import Dataset

class AcousticDataset(Dataset):

    def __init__(
        self,
        feature_root,
        metadata_df,
        feature_type,
        preload=True
    ):

        self.feature_root = feature_root

        self.filenames = (
            metadata_df["filename"]
            .tolist()
        )

        self.labels = (
            metadata_df["label"]
            .tolist()
        )

        # =====================================
        # Feature type
        # =====================================

        if isinstance(feature_type, str):
            self.feature_type = [feature_type]
        else:
            self.feature_type = feature_type

        # =====================================
        # Build paths
        # =====================================

        self.feature_paths = []

        for filename in self.filenames:

            paths = []

            for feat in self.feature_type:

                path = os.path.join(
                    self.feature_root,
                    feat,
                    filename
                )

                paths.append(path)

            self.feature_paths.append(paths)

        # =====================================
        # PRELOAD TO RAM
        # =====================================

        self.preload = preload

        if self.preload:

            print("Preloading features into RAM...")

            self.cache = []

            for paths in self.feature_paths:

                feature_list = []

                for path in paths:

                    x = np.load(path)

                    if x.ndim == 1:
                        x = x[:, np.newaxis]

                    feature_list.append(x)

                # early fusion
                x = np.concatenate(
                    feature_list,
                    axis=0
                )

                # add channel dimension
                x = np.expand_dims(
                    x,
                    axis=0
                )

                # numpy -> tensor
                x = torch.from_numpy(x).float()

                self.cache.append(x)

            print("Preload finished.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        # =====================================
        # LOAD FROM RAM CACHE
        # =====================================

        if self.preload:

            x = self.cache[idx]

        # =====================================
        # LOAD FROM DISK
        # =====================================

        else:

            feature_list = []

            for path in self.feature_paths[idx]:

                x = np.load(path)

                if x.ndim == 1:
                    x = x[:, np.newaxis]

                feature_list.append(x)

            x = np.concatenate(
                feature_list,
                axis=0
            )

            x = np.expand_dims(
                x,
                axis=0
            )

            x = torch.from_numpy(x).float()

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return x, y