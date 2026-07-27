import torch
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)

from cnn import AudioCNN
from Scripts.Data_Loader.acoustic_dataloader import AcousticDataset

def main():
    # =========================================================
    # CONFIG
    # =========================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 20

    BASE_PATH = "Dataset/CREMAD"

    FEATURE_ROOT = os.path.join(
        BASE_PATH,
        "Acoustic_Features"
    )

    SPLIT_PATH = os.path.join(
        BASE_PATH,
        "Split/Chyan/train_test"
    )

    SAVE_DIR = "Results/Tuning"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================================================
    # FEATURE COMBINATIONS
    # =========================================================

    FEATURES = [
        ["MFCC"],
        ["GFCC"],
        ["LPC"],
        ["MFCC", "GFCC"],
        ["MFCC", "LPC"],
        ["GFCC", "LPC"],
        ["MFCC", "GFCC", "LPC"]
    ]

    # =========================================================
    # HYPERPARAMETER GRID
    # =========================================================

    param_grid = {
        "k1": [32, 64, 128],
        "k2": [64, 128, 256],
        "dropout1": [0.2, 0.3],
        "dropout2": [0.3, 0.4],
        "batch_size": [16, 32, 64],
        "lr": [1e-2, 1e-3, 1e-4],
    }

    # =========================================================
    # LOAD METADATA
    # =========================================================

    metadata = pd.read_csv(
        os.path.join(
            SPLIT_PATH,
            "metadata.csv"
        )
    )

    # =========================================================
    # LOAD TRAIN TEST SPLIT
    # =========================================================

    split = np.load(
        os.path.join(
            SPLIT_PATH,
            "train_test_80_20_group_actor.npz"
        )
    )

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    # =========================================================
    # STORAGE
    # =========================================================

    results = []

    experiment_id = 0

    # =========================================================
    # GRID SEARCH
    # =========================================================

    for feature_list in tqdm(
        FEATURES,
        desc="Feature Grid Search"
    ):

        feature_name = "+".join(feature_list)

        print(f"\nGRID SEARCH FEATURE: {feature_name}")

        # -----------------------------------------------------
        # DATASET
        # -----------------------------------------------------

        dataset = AcousticDataset(
            FEATURE_ROOT,
            metadata,
            feature_type=feature_list
        )

        # =====================================================
        # PARAMETER COMBINATION LOOP
        # =====================================================

        for values in product(*param_grid.values()):

            experiment_id += 1

            cfg = dict(
                zip(param_grid.keys(), values)
            )

            print(f"\nExperiment {experiment_id}")
            print(f"Config: {cfg}")

            # -------------------------------------------------
            # DATALOADER
            # -------------------------------------------------

            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=cfg["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

            test_loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

            # -------------------------------------------------
            # MODEL
            # -------------------------------------------------

            model = AudioCNN(
                num_classes=metadata["label"].nunique(),
                k1=cfg["k1"],
                k2=cfg["k2"],
                dropout1=cfg["dropout1"],
                dropout2=cfg["dropout2"]
            ).to(device)

            # -------------------------------------------------
            # PARAMETER COUNT
            # -------------------------------------------------

            num_params = sum(
                p.numel()
                for p in model.parameters()
            )

            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg["lr"]
            )

            criterion = nn.CrossEntropyLoss()

            # =================================================
            # TRAINING
            # =================================================

            train_losses = []

            start_time = time.time()

            for epoch in range(EPOCHS):

                model.train()

                epoch_loss = 0.0

                for x, y in train_loader:

                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()

                    outputs = model(x)

                    loss = criterion(outputs, y)

                    loss.backward()

                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)

                train_losses.append(avg_loss)

                # print(
                #     f"Epoch [{epoch+1}/{EPOCHS}] "
                #     f"Loss: {avg_loss:.4f}"
                # )

            training_time = time.time() - start_time

            # =================================================
            # TESTING
            # =================================================

            model.eval()

            y_true = []
            y_pred = []

            with torch.no_grad():

                for x, y in test_loader:

                    x = x.to(device)

                    pred = (
                        model(x)
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                    ) 

                    y_pred.extend(pred)

                    y_true.extend(y.numpy())

            # =================================================
            # METRICS
            # =================================================

            test_acc = accuracy_score(
                y_true,
                y_pred
            ) * 100

            precision = precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )

            # -------------------------------------------------
            # CONFUSION MATRIX
            # -------------------------------------------------

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            cm_path = os.path.join(
                SAVE_DIR,
                f"cm_exp_{experiment_id}.csv"
            )

            pd.DataFrame(cm).to_csv(
                cm_path,
                index=False
            )

            # -------------------------------------------------
            # CLASSIFICATION REPORT
            # -------------------------------------------------

            report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            report_path = os.path.join(
                SAVE_DIR,
                f"report_exp_{experiment_id}.csv"
            )

            report_df.to_csv(report_path)

            # -------------------------------------------------
            # SAVE LEARNING CURVE
            # -------------------------------------------------

            learning_curve_df = pd.DataFrame({
                "epoch": np.arange(1, EPOCHS + 1),
                "train_loss": train_losses
            })

            curve_path = os.path.join(
                SAVE_DIR,
                f"learning_curve_exp_{experiment_id}.csv"
            )

            learning_curve_df.to_csv(
                curve_path,
                index=False
            )

            # =================================================
            # SAVE RESULT
            # =================================================

            results.append({

                # -----------------------------
                # experiment info
                # -----------------------------

                "experiment_id": experiment_id,

                # -----------------------------
                # feature
                # -----------------------------

                "feature": feature_name,

                # -----------------------------
                # hyperparameter
                # -----------------------------

                **cfg,

                # -----------------------------
                # metrics
                # -----------------------------

                "test_acc": test_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,

                # -----------------------------
                # training info
                # -----------------------------

                "final_train_loss": train_losses[-1],
                "min_train_loss": np.min(train_losses),
                "training_time_sec": training_time,

                # -----------------------------
                # model complexity
                # -----------------------------

                "num_params": num_params,

                # -----------------------------
                # saved files
                # -----------------------------

                "confusion_matrix_path": cm_path,
                "classification_report_path": report_path,
                "learning_curve_path": curve_path
            })

            print(f"Test Accuracy : {test_acc:.2f}%")
            print(f"Precision     : {precision:.4f}")
            print(f"Recall        : {recall:.4f}")
            print(f"F1 Score      : {f1:.4f}")
            print(f"Train Time    : {training_time:.2f} sec")
            print(f"Parameters    : {num_params:,}")

    # =========================================================
    # SAVE ALL RESULTS
    # =========================================================

    df_results = pd.DataFrame(results)

    csv_result_path = os.path.join(
        SAVE_DIR,
        "cnn_results.csv"
    )

    df_results.to_csv(
        csv_result_path,
        index=False
    )

    # =========================================================
    # TOP CONFIGURATION
    # =========================================================

    print("\nTOP 10 CONFIGURATIONS")

    print(
        df_results
        .sort_values(
            "test_acc",
            ascending=False
        )
        .head(10)
    )

    print(f"\nResults saved to: {csv_result_path}")
    
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()