import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
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

    EPOCHS = 100
    N_FOLDS = 5

    BASE_PATH = "Dataset/CREMAD"

    FEATURE_ROOT = os.path.join(
        BASE_PATH,
        "Acoustic_Features"
    )

    KFOLD_PATH = os.path.join(
        BASE_PATH,
        "Split/Chyan/kfold"
    )

    CSV_PATH = "Results/Tuning/cnn_results.csv"

    SAVE_DIR = "Results/Final_KFold"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================================================
    # BASELINE CONFIG
    # =========================================================

    BASELINE_CONFIG = {
        "k1": 32,
        "k2": 64,
        "dropout1": 0.2,
        "dropout2": 0.3,
        "batch_size": 32,
        "lr": 0.001
    }

    # =========================================================
    # LOAD BEST CONFIG
    # =========================================================

    df = pd.read_csv(CSV_PATH)

    best_configs = (
        df.sort_values(
            "f1_score",
            ascending=False
        )
        .groupby("feature")
        .head(1)
        .reset_index(drop=True)
    )

    print("\nBEST CONFIG PER FEATURE")
    print(best_configs)

    # =========================================================
    # LOAD METADATA
    # =========================================================

    metadata = pd.read_csv(
        os.path.join(KFOLD_PATH, "metadata.csv")
    )

    label_names = sorted(metadata["label"].unique())

    # =========================================================
    # STORAGE
    # =========================================================

    final_results = []

    all_predictions = []

    # =========================================================
    # LOOP FEATURE
    # =========================================================

    for _, row in tqdm(
        best_configs.iterrows(),
        total=len(best_configs),
        desc="Final KFold Training"
    ):

        feature = row["feature"]

        # =====================================================
        # FEATURE LIST
        # =====================================================

        if "+" in feature:
            feature_list = feature.split("+")
        else:
            feature_list = [feature]

        # =====================================================
        # CONFIGURATION LIST
        # =====================================================

        tuned_config = {
            "k1": int(row["k1"]),
            "k2": int(row["k2"]),
            "dropout1": float(row["dropout1"]),
            "dropout2": float(row["dropout2"]),
            "batch_size": int(row["batch_size"]),
            "lr": float(row["lr"]),
        }

        config_list = [
            ("baseline", BASELINE_CONFIG),
            ("tuned", tuned_config)
        ]

        # =====================================================
        # DATASET
        # =====================================================

        dataset = AcousticDataset(
            FEATURE_ROOT,
            metadata,
            feature_type=feature_list
        )

        # =====================================================
        # LOOP CONFIG
        # =====================================================

        for config_name, CONFIG in config_list:

            print("\n=================================================")
            print(f"Feature   : {feature}")
            print(f"Config    : {config_name}")
            print(f"Parameter : {CONFIG}")
            print("=================================================")

            # =================================================
            # STORAGE PER CONFIG
            # =================================================

            fold_accuracies = []
            fold_f1_scores = []
            fold_precision_scores = []
            fold_recall_scores = []

            feature_train_losses = []
            feature_val_losses = []

            feature_train_accuracies = []
            feature_val_accuracies = []

            feature_val_f1 = []

            total_cm = np.zeros(
                (
                    len(label_names),
                    len(label_names)
                ),
                dtype=int
            )

            # =================================================
            # K-FOLD
            # =================================================

            for fold in range(1, N_FOLDS + 1):

                print(f"\nFold {fold}")

                split = np.load(
                    os.path.join(
                        KFOLD_PATH,
                        f"fold_{fold}.npz"
                    )
                )

                train_idx = split["train_idx"]
                val_idx = split["val_idx"]

                # =============================================
                # DATALOADER
                # =============================================

                train_loader = DataLoader(
                    Subset(dataset, train_idx),
                    batch_size=CONFIG["batch_size"],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )

                val_loader = DataLoader(
                    Subset(dataset, val_idx),
                    batch_size=CONFIG["batch_size"],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )

                # =============================================
                # MODEL
                # =============================================

                model = AudioCNN(
                    num_classes=metadata["label"].nunique(),
                    k1=CONFIG["k1"],
                    k2=CONFIG["k2"],
                    dropout1=CONFIG["dropout1"],
                    dropout2=CONFIG["dropout2"]
                ).to(device)

                optimizer = optim.Adam(
                    model.parameters(),
                    lr=CONFIG["lr"]
                )

                criterion = nn.CrossEntropyLoss()

                # =============================================
                # PARAMETER COUNT
                # =============================================

                num_params = sum(
                    p.numel()
                    for p in model.parameters()
                )

                # =============================================
                # TRAINING
                # =============================================

                start_time = time.time()

                train_losses = []
                val_losses = []

                train_acc_epochs = []
                val_acc_epochs = []

                val_f1_epochs = []

                for epoch in range(EPOCHS):

                    # =========================================
                    # TRAIN
                    # =========================================

                    model.train()

                    epoch_loss = 0.0
                    
                    train_y_true = []
                    train_y_pred = []

                    for x, y in train_loader:

                        x = x.to(device)
                        y = y.to(device)

                        optimizer.zero_grad()

                        outputs = model(x)

                        loss = criterion(outputs, y)

                        probs = torch.softmax(outputs, dim=1)
                        pred = probs.argmax(dim=1)

                        train_y_pred.extend(pred.cpu().numpy())
                        train_y_true.extend(y.cpu().numpy())

                        loss.backward()

                        optimizer.step()

                        epoch_loss += loss.item()

                    avg_loss = epoch_loss / len(train_loader)

                    train_acc = accuracy_score(
                        train_y_true,
                        train_y_pred
                    ) * 100

                    train_losses.append(avg_loss)
                    train_acc_epochs.append(train_acc)

                    # =========================================
                    # VALIDATION
                    # =========================================

                    model.eval()

                    epoch_y_true = []

                    epoch_y_pred = []

                    val_loss = 0.0


                    with torch.no_grad():

                        for x, y in val_loader:

                            x = x.to(device)
                            y = y.to(device)

                            outputs = model(x)

                            loss = criterion(outputs, y)

                            val_loss += loss.item()

                            probs = torch.softmax(outputs, dim=1)
                            pred = probs.argmax(dim=1).cpu().numpy()

                            epoch_y_pred.extend(pred)
                            epoch_y_true.extend(y.cpu().numpy())
                            
                    epoch_acc = accuracy_score(
                        epoch_y_true,
                        epoch_y_pred
                    ) * 100

                    epoch_f1 = f1_score(
                        epoch_y_true,
                        epoch_y_pred,
                        average="macro"
                    )

                    val_acc_epochs.append(epoch_acc)

                    val_f1_epochs.append(epoch_f1)
                    
                    avg_val_loss = val_loss / len(val_loader)

                    val_losses.append(avg_val_loss)

                    print(
                        f"Epoch [{epoch+1}/{EPOCHS}] "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f}% | "
                        f"Val Acc: {epoch_acc:.2f}% | "
                        f"Val F1: {epoch_f1:.4f}"
                    )
                    
                    

                elapsed_time = time.time() - start_time

                # =============================================
                # FINAL VALIDATION
                # =============================================

                model.eval()

                y_true = []

                y_pred = []

                
                with torch.no_grad():

                    for x, y in val_loader:

                        x = x.to(device)

                        pred = (
                            model(x)
                            .argmax(dim=1)
                            .cpu()
                            .numpy()
                        )

                        y_pred.extend(pred)

                        y_true.extend(y.cpu().numpy())

                # =============================================
                # METRICS
                # =============================================

                acc = accuracy_score(
                    y_true,
                    y_pred
                ) * 100

                f1 = f1_score(
                    y_true,
                    y_pred,
                    average="macro"
                )

                precision = precision_score(
                    y_true,
                    y_pred,
                    average="macro"
                )

                recall = recall_score(
                    y_true,
                    y_pred,
                    average="macro"
                )

                fold_accuracies.append(acc)

                fold_f1_scores.append(f1)

                fold_precision_scores.append(precision)

                fold_recall_scores.append(recall)

                print(f"Fold {fold} Accuracy : {acc:.2f}%")
                print(f"Fold {fold} F1-Score: {f1:.4f}")
                
                # =============================================
                # SAVE MODEL
                # =============================================

                model_path = os.path.join(
                    SAVE_DIR,
                    f"{feature}_{config_name}_fold{fold}_model.pth"
                )

                torch.save({
                    "feature": feature,
                    "config_type": config_name,
                    "fold": fold,

                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),

                    "accuracy": acc,
                    "f1_score": f1,

                    "config": CONFIG,

                    "label_names": label_names
                }, model_path)

                print(f"Model saved: {model_path}")

                # =============================================
                # SAVE CURVE DATA
                # =============================================

                # SAVE CURVE DATA

                feature_train_losses.append(train_losses)
                feature_val_losses.append(val_losses)

                feature_train_accuracies.append(train_acc_epochs)
                feature_val_accuracies.append(val_acc_epochs)

                feature_val_f1.append(val_f1_epochs)

                # =============================================
                # CONFUSION MATRIX
                # =============================================

                cm = confusion_matrix(
                    y_true,
                    y_pred
                )

                total_cm += cm

                # =============================================
                # CLASSIFICATION REPORT
                # =============================================

                report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True
                )

                report_path = os.path.join(
                    SAVE_DIR,
                    f"{feature}_{config_name}_fold{fold}_classification_report.json"
                )

                with open(report_path, "w") as f:
                    json.dump(report, f, indent=4)

                # =============================================
                # SAVE PREDICTION
                # =============================================

                for t, p in zip(y_true, y_pred):

                    all_predictions.append({
                        "feature": feature,
                        "config": config_name,
                        "fold": fold,
                        "true_label": int(t),
                        "pred_label": int(p)
                    })

            # =================================================
            # FINAL METRIC
            # =================================================

            mean_acc = np.mean(fold_accuracies)

            std_acc = np.std(fold_accuracies)

            mean_f1 = np.mean(fold_f1_scores)

            std_f1 = np.std(fold_f1_scores)

            mean_precision = np.mean(
                fold_precision_scores
            )

            mean_recall = np.mean(
                fold_recall_scores
            )

            print("\n=================================================")
            print(f"{feature} ({config_name})")
            print(f"Mean Accuracy : {mean_acc:.2f}% ± {std_acc:.2f}")
            print(f"Mean F1-Score : {mean_f1:.4f} ± {std_f1:.4f}")
            print("=================================================")

            # =================================================
            # SAVE CONFUSION MATRIX
            # =================================================

            cm_df = pd.DataFrame(
                total_cm,
                index=label_names,
                columns=label_names
            )

            cm_df.to_csv(
                os.path.join(
                    SAVE_DIR,
                    f"{feature}_{config_name}_confusion_matrix.csv"
                )
            )

            # =================================================
            # SAVE LEARNING CURVE
            # =================================================

            learning_curve_df = pd.DataFrame({
                "epoch": np.arange(1, EPOCHS + 1),

                "train_loss": np.mean(
                    feature_train_losses,
                    axis=0
                ),

                "val_loss": np.mean(
                    feature_val_losses,
                    axis=0
                ),

                "train_acc": np.mean(
                    feature_train_accuracies,
                    axis=0
                ),

                "val_acc": np.mean(
                    feature_val_accuracies,
                    axis=0
                ),

                "val_f1": np.mean(
                    feature_val_f1,
                    axis=0
                )
            })
            
            learning_curve_df.to_csv(
                os.path.join(
                    SAVE_DIR,
                    f"{feature}_{config_name}_learning_curve.csv"
                ),
                index=False
            )

            # =================================================
            # SAVE FINAL RESULT
            # =================================================

            result = {
                "feature": feature,
                "config_type": config_name,

                "mean_acc": mean_acc,
                "std_acc": std_acc,

                "mean_f1": mean_f1,
                "std_f1": std_f1,

                "mean_precision": mean_precision,
                "mean_recall": mean_recall,

                "num_params": num_params,
                "training_time_sec": elapsed_time,

                **CONFIG
            }

            # =============================================
            # SAVE FOLD RESULT
            # =============================================

            for i in range(N_FOLDS):

                result[f"fold_{i+1}_acc"] = (
                    fold_accuracies[i]
                )

                result[f"fold_{i+1}_f1"] = (
                    fold_f1_scores[i]
                )

            final_results.append(result)

    # =========================================================
    # SAVE FINAL RESULTS
    # =========================================================

    df_final = pd.DataFrame(final_results)

    df_final.to_csv(
        os.path.join(
            SAVE_DIR,
            "final_kfold_results.csv"
        ),
        index=False
    )

    # =========================================================
    # SAVE PREDICTIONS
    # =========================================================

    pred_df = pd.DataFrame(all_predictions)

    pred_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "all_predictions.csv"
        ),
        index=False
    )

    print("\nFINAL RESULTS SAVED")
    print("Location:", SAVE_DIR)


if __name__ == "__main__":

    torch.multiprocessing.freeze_support()

    main()