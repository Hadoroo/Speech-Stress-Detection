import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from Scripts.Data_Collector.temporal_feature_collector import TemporalFeatures

# ---------------- Feature Types ----------------
features = ["MFCC", "GFCC", "LogFBank", "f0"]

# ---------------- Improved SVC Grid Search ----------------
def improved_svc_grid_search(features):
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    save_dir = "Results/Tuning/SVC_Classifier"
    os.makedirs(save_dir, exist_ok=True)

    for feat in features:
        save_path = os.path.join(save_dir, f"{feat}_svc.npy")
        if os.path.exists(save_path):
            print(f"⏩ Skip {feat}, tuning exists: {save_path}")
            continue

        print(f"\n🔎 Grid search untuk fitur: {feat}")

        train_ds = TemporalFeatures(model_state="train", acoustic_feature_type=feat, temporal_feature_type="summary")
        test_ds  = TemporalFeatures(model_state="test",  acoustic_feature_type=feat, temporal_feature_type="summary")

        print("📥 Mengonversi train dataset ke NumPy...")
        X_train = np.array([train_ds[i][0].numpy().flatten() for i in tqdm(range(len(train_ds)), desc=f"{feat} Train")])
        y_train = np.array([train_ds[i][1] for i in tqdm(range(len(train_ds)), desc=f"{feat} Label Train")])
        print("📥 Mengonversi test dataset ke NumPy...")
        X_test  = np.array([test_ds[i][0].numpy().flatten() for i in tqdm(range(len(test_ds)), desc=f"{feat} Test")])
        y_test  = np.array([test_ds[i][1] for i in tqdm(range(len(test_ds)), desc=f"{feat} Label Test")])

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True))
        ])

        print(f"⚙️ Melakukan grid search untuk {feat} ...")
        grid_search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=0)
        with tqdm(total=len(param_grid['svc__C']) * len(param_grid['svc__kernel']) * len(param_grid['svc__gamma']),
                  desc=f"GridSearch {feat}", unit="comb") as pbar:
            grid_search.fit(X_train, y_train)
            pbar.update(pbar.total)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        best_acc = accuracy_score(y_test, y_pred)
        best_f1 = f1_score(y_test, y_pred, average='weighted')

        result = {
            'best_params': grid_search.best_params_,
            'best_acc': best_acc,
            'best_f1': best_f1
        }
        np.save(save_path, result)

        print(f"✅ Best params for {feat}: {grid_search.best_params_}")
        print(f"✅ Validation accuracy: {best_acc:.4f}, F1-score: {best_f1:.4f}")

# ---------------- Improved Train and Evaluate ----------------
def improved_svc_train_and_evaluate(features):
    model_dir = "Results/Models/SVC"
    os.makedirs(model_dir, exist_ok=True)
    results_dir = "Results/Predictions/SVC"
    os.makedirs(results_dir, exist_ok=True)

    for feat in features:
        tune_path = f"Results/Tuning/SVC_Classifier/{feat}_svc.npy"
        if not os.path.exists(tune_path):
            print(f"⏭️ Skip {feat}, tuning file not found: {tune_path}")
            continue

        tune_result = np.load(tune_path, allow_pickle=True).item()
        best_params = tune_result['best_params']

        print(f"\n🚀 Training final SVC for {feat} with params: {best_params}")

        train_ds = TemporalFeatures(model_state="train", acoustic_feature_type=feat, temporal_feature_type="summary")
        test_ds  = TemporalFeatures(model_state="test",  acoustic_feature_type=feat, temporal_feature_type="summary")

        print("📥 Mengonversi dataset...")
        X_train = np.array([train_ds[i][0].numpy().flatten() for i in tqdm(range(len(train_ds)), desc=f"{feat} Train")])
        y_train = np.array([train_ds[i][1] for i in tqdm(range(len(train_ds)), desc=f"{feat} Label Train")])
        X_test  = np.array([test_ds[i][0].numpy().flatten() for i in tqdm(range(len(test_ds)), desc=f"{feat} Test")])
        y_test  = np.array([test_ds[i][1] for i in tqdm(range(len(test_ds)), desc=f"{feat} Label Test")])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("🏋️‍♂️ Training SVC model...")
        svc = SVC(
            C=best_params['svc__C'],
            kernel=best_params['svc__kernel'],
            gamma=best_params['svc__gamma'],
            probability=True
        )

        svc.fit(X_train_scaled, y_train)

        # ---------------- EVALUATION ----------------
        y_pred_train = svc.predict(X_train_scaled)
        y_pred_test = svc.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Metrik tambahan
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        cm = confusion_matrix(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test, output_dict=True)

        results = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'params': best_params
        }

        np.save(os.path.join(results_dir, f"{feat}_svc_results.npy"), results)

        print(f"\n📊 Final Results for {feat}:")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"✅ SVC training & evaluation for {feat} completed!")

# ---------------- Run Pipeline ----------------
if __name__ == "__main__":
    print("🚀 Starting SVC Training Pipeline...")
    improved_svc_grid_search(features)
    improved_svc_train_and_evaluate(features)
    print("🎉 SVC pipeline completed!")
