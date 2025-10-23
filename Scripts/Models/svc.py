import os, sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hinge_loss
)
from sklearn.exceptions import ConvergenceWarning
from Scripts.Data_Collector.temporal_feature_collector import TemporalFeatures

# ==========================================================
#                    KONFIGURASI FITUR & DATASET
# ==========================================================
features = ["MFCC", "GFCC", "LogFBank", "F0"]
datasets = ["CREMAD", "RAVDESS", "TESS"]

# ==========================================================
#                IGNORE SVC CONVERGENCE WARNING
# ==========================================================
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==========================================================
#                    GRID SEARCH (IMPROVED)
# ==========================================================
def improved_svc_grid_search(features, datasets):
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf', 'poly'],
        'svc__gamma': ['scale', 'auto', 0.01],
        'svc__max_iter': [1000, 5000, 10000]
    }


    save_dir = "Results/Tuning/SVC_Classifier"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"\n📂 Dataset: {dataset_name}")
        for feat in features:
            save_path = os.path.join(save_dir, f"{dataset_name}_{feat}_svc.npy")
            if os.path.exists(save_path):
                print(f"⏩ Skip {feat}, tuning exists: {save_path}")
                continue

            print(f"\n🔎 Melakukan grid search untuk fitur: {feat}")

            # Load dataset
            train_ds = TemporalFeatures(dataset_name=dataset_name, model_state="train",
                                        acoustic_feature_type=feat, temporal_feature_type="summary")
            test_ds = TemporalFeatures(dataset_name=dataset_name, model_state="test",
                                       acoustic_feature_type=feat, temporal_feature_type="summary")

            # Convert ke NumPy
            print("📥 Mengonversi dataset ke NumPy...")
            X_train = np.array([train_ds[i][0].numpy().flatten() for i in tqdm(range(len(train_ds)), desc=f"{dataset_name}-{feat} Train")])
            y_train = np.array([train_ds[i][1] for i in tqdm(range(len(train_ds)), desc=f"{dataset_name}-{feat} Label Train")])
            X_test = np.array([test_ds[i][0].numpy().flatten() for i in tqdm(range(len(test_ds)), desc=f"{dataset_name}-{feat} Test")])
            y_test = np.array([test_ds[i][1] for i in tqdm(range(len(test_ds)), desc=f"{dataset_name}-{feat} Label Test")])

            # Pipeline: StandardScaler + SVC
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(probability=True, max_iter=10000))
            ])

            print(f"⚙️ Melakukan Grid SearchCV untuk {feat} ({dataset_name}) ...")
            grid_search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Evaluasi metrik
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            result = {
                'best_params': grid_search.best_params_,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist()
            }

            np.save(save_path, result)
            print(f"✅ Best params for {dataset_name}-{feat}: {grid_search.best_params_}")
            print(f"📊 Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

# ==========================================================
#             FINAL TRAIN & EVALUATE (IMPROVED)
# ==========================================================
def improved_svc_train_and_evaluate(features, datasets):
    model_dir = "Results/Models/SVC"
    os.makedirs(model_dir, exist_ok=True)
    results_dir = "Results/Predictions/SVC"
    os.makedirs(results_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"\n📂 Dataset: {dataset_name}")
        for feat in features:
            tune_path = f"Results/Tuning/SVC_Classifier/{dataset_name}_{feat}_svc.npy"
            if not os.path.exists(tune_path):
                print(f"⏭️ Skip {feat}, tuning file not found: {tune_path}")
                continue

            tune_result = np.load(tune_path, allow_pickle=True).item()
            best_params = tune_result['best_params']

            print(f"\n🚀 Training final SVC for {dataset_name}-{feat} with params: {best_params}")

            # Load dataset ulang
            train_ds = TemporalFeatures(model_state="train", dataset_name=dataset_name,
                                        acoustic_feature_type=feat, temporal_feature_type="summary")
            test_ds = TemporalFeatures(model_state="test", dataset_name=dataset_name,
                                       acoustic_feature_type=feat, temporal_feature_type="summary")

            # Convert ke NumPy
            X_train = np.array([train_ds[i][0].numpy().flatten() for i in range(len(train_ds))])
            y_train = np.array([train_ds[i][1] for i in range(len(train_ds))])
            X_test = np.array([test_ds[i][0].numpy().flatten() for i in range(len(test_ds))])
            y_test = np.array([test_ds[i][1] for i in range(len(test_ds))])

            # Standarisasi
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model akhir
            svc = SVC(
                C=best_params['svc__C'],
                kernel=best_params['svc__kernel'],
                gamma=best_params['svc__gamma'],
                degree=best_params.get('svc__degree', 3),
                coef0=best_params.get('svc__coef0', 0.0),
                class_weight=best_params.get('svc__class_weight', None),
                shrinking=best_params.get('svc__shrinking', True),
                tol=best_params.get('svc__tol', 1e-3),
                max_iter=best_params.get('svc__max_iter', 10000),
                decision_function_shape=best_params.get('svc__decision_function_shape', 'ovr'),
                probability=True
            )

            svc.fit(X_train_scaled, y_train)
            y_pred_train = svc.predict(X_train_scaled)
            y_pred_test = svc.predict(X_test_scaled)

            # Evaluasi metrik
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred_test)
            loss_train = hinge_loss(y_train, svc.decision_function(X_train_scaled))
            loss_test = hinge_loss(y_test, svc.decision_function(X_test_scaled))

            results = {
                'dataset': dataset_name,
                'feature': feat,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'loss_train': loss_train,
                'loss_test': loss_test,
                'confusion_matrix': cm.tolist(),
                'params': best_params
            }

            np.save(os.path.join(results_dir, f"{dataset_name}_{feat}_svc_results.npy"), results)

            print(f"✅ SVC training & evaluation for {dataset_name}-{feat} completed!")

# ==========================================================
#                    RUN PIPELINE
# ==========================================================
if __name__ == "__main__":
    print("🚀 Starting Multi-Dataset SVC Training Pipeline...")
    improved_svc_grid_search(features, datasets)
    improved_svc_train_and_evaluate(features, datasets)
    print("🎉 SVC pipeline for all datasets completed!")
