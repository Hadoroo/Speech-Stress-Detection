import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Scripts.Data_Collector.temporal_feature_collector import TemporalFeatures

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------------- Feature Types ----------------
features = ["MFCC", "GFCC", "LogFBank", "F0"]

# ---------------- Dataset List ----------------
datasets = ["CREMAD", "RAVDESS", "TESS"]

# ---------------- Improved Collate Function ----------------
def collate_fn(batch):
    feats, labels, fnames = zip(*batch)
    feats = torch.stack(feats)
    labels = torch.tensor(labels, dtype=torch.long)
    return feats, labels, fnames

# ---------------- Improved VGG16 Model ----------------
class ImprovedVGG16Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout_rate=0.5, hidden_dim=512):
        super().__init__()
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim // 4)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(hidden_dim // 8, num_classes)
        )

    def forward(self, x):
        x = self.feature_processor(x)
        x = self.layer_norm(x)
        x = self.classifier(x)
        return x

# ---------------- Class Weights ----------------
def calculate_class_weights(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ---------------- Safe Batch Training ----------------
def safe_batch_training(model, train_loader, criterion, optimizer, epoch_idx=None, total_epochs=None):
    model.train()
    train_loss = 0
    
    # Handle optional epoch parameters for progress bar
    if epoch_idx is not None and total_epochs is not None:
        desc = f"Epoch {epoch_idx+1}/{total_epochs} [Train]"
    else:
        desc = "Training"
        
    progress_bar = tqdm(train_loader, desc=desc, leave=False)
    for X, y, _ in progress_bar:
        if X.size(0) == 1:
            continue
        X, y = X.to(device), y.to(device)
        X = X.view(X.size(0), -1).float()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    progress_bar.close()
    return train_loss / max(len(train_loader), 1)

# ---------------- Quick Training (for Grid Search) ----------------
def quick_training(model, train_loader, criterion, optimizer, num_epochs=3):
    """Simplified training function for grid search without progress bar details"""
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y, _ in train_loader:
            if X.size(0) == 1:
                continue
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1).float()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
    return epoch_loss / max(len(train_loader), 1)

# ---------------- Evaluation (with metrics) ----------------
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y, _ in dataloader:
            if X.size(0) == 1:
                continue
            X, y = X.to(device), y.to(device)
            X = X.view(X.size(0), -1).float()
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    if len(all_labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------------- Grid Search ----------------
def improved_grid_search(features, datasets):
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [16, 32],
        'hidden_dim': [256, 512],
        'dropout_rate': [0.3, 0.5],
        'weight_decay': [1e-5, 0.0]
    }

    save_dir = "Results/Tuning/VGG16_Classifier"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"\n📂 Dataset: {dataset_name}")
        for feat in features:
            save_path = os.path.join(save_dir, f"{dataset_name}_{feat}_vgg16.npy")
            if os.path.exists(save_path):
                print(f"⏩ Skip {feat}, tuning exists: {save_path}")
                continue

            print(f"\n🔎 Grid search untuk {feat} ({dataset_name})")

            train_ds = TemporalFeatures(dataset_name=dataset_name, model_state="train",
                                        acoustic_feature_type=feat, temporal_feature_type="summary")
            test_ds = TemporalFeatures(dataset_name=dataset_name, model_state="test",
                                       acoustic_feature_type=feat, temporal_feature_type="summary")

            sample_feat, _, _ = train_ds[0]
            input_dim = sample_feat.numel()

            best_acc, best_params = 0.0, None
            for params in tqdm(list(ParameterGrid(param_grid)), desc=f"Tuning {feat}"):
                lr, bs, hidden_dim = params['learning_rate'], params['batch_size'], params['hidden_dim']
                dropout_rate, weight_decay = params['dropout_rate'], params['weight_decay']

                actual_bs = min(bs, len(train_ds))
                if actual_bs < 2: continue

                train_loader = DataLoader(train_ds, batch_size=actual_bs, shuffle=True, collate_fn=collate_fn)
                test_loader = DataLoader(test_ds, batch_size=actual_bs, shuffle=False, collate_fn=collate_fn)

                model = ImprovedVGG16Classifier(input_dim, 2, dropout_rate, hidden_dim).to(device)
                class_weights = calculate_class_weights(train_ds)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

                # Use quick_training instead of safe_batch_training for grid search
                quick_training(model, train_loader, criterion, optimizer, num_epochs=3)

                val_metrics = evaluate_model(model, test_loader)
                val_acc = val_metrics["accuracy"]
                if val_acc > best_acc:
                    best_acc, best_params = val_acc, params

            if best_params:
                np.save(save_path, {'params': best_params, 'best_acc': best_acc, 'input_dim': input_dim})
                print(f"✅ Best for {dataset_name}-{feat}: {best_params} (acc={best_acc:.4f})")
            else:
                print(f"❌ No valid params for {dataset_name}-{feat}")

# ---------------- Train & Evaluate ----------------
def improved_train_and_evaluate(features, datasets):
    model_dir = "Results/Models/VGG16_Improved"
    os.makedirs(model_dir, exist_ok=True)
    results_dir = "Results/Predictions/VGG16_Improved"
    os.makedirs(results_dir, exist_ok=True)

    for dataset_name in datasets:
        print(f"\n📂 Dataset: {dataset_name}")
        for feat in features:
            tune_path = f"Results/Tuning/VGG16_Classifier/{dataset_name}_{feat}_vgg16.npy"
            if not os.path.exists(tune_path):
                print(f"⏭️ Skip {feat}, tuning file not found.")
                continue

            tune_result = np.load(tune_path, allow_pickle=True).item()
            best_params = tune_result['params']
            input_dim = tune_result['input_dim']

            lr, bs, hidden_dim = best_params['learning_rate'], best_params['batch_size'], best_params['hidden_dim']
            dropout_rate, weight_decay = best_params['dropout_rate'], best_params['weight_decay']

            print(f"\n🚀 Training {dataset_name}-{feat} with {best_params}")

            train_ds = TemporalFeatures(dataset_name=dataset_name, model_state="train",
                                        acoustic_feature_type=feat, temporal_feature_type="summary")
            test_ds = TemporalFeatures(dataset_name=dataset_name, model_state="test",
                                       acoustic_feature_type=feat, temporal_feature_type="summary")

            actual_bs = max(2, min(bs, len(train_ds)))
            train_loader = DataLoader(train_ds, batch_size=actual_bs, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_ds, batch_size=actual_bs, shuffle=False, collate_fn=collate_fn)

            model = ImprovedVGG16Classifier(input_dim, 2, dropout_rate, hidden_dim).to(device)
            class_weights = calculate_class_weights(train_ds)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

            history = {"train_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []}

            for epoch in range(30):
                train_loss = safe_batch_training(model, train_loader, criterion, optimizer, epoch, 30)
                val_metrics = evaluate_model(model, test_loader)
                scheduler.step(val_metrics["accuracy"])

                history["train_loss"].append(train_loss)
                history["val_acc"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])

                tqdm.write(
                    f"Epoch {epoch+1:02d} | Loss={train_loss:.4f} | "
                    f"Acc={val_metrics['accuracy']:.4f} | Prec={val_metrics['precision']:.4f} | "
                    f"Rec={val_metrics['recall']:.4f} | F1={val_metrics['f1']:.4f}"
                )

            # Save final model & results
            torch.save(model.state_dict(), os.path.join(model_dir, f"{dataset_name}_{feat}_final.pt"))
            test_metrics = evaluate_model(model, test_loader)

            results = {
                'dataset': dataset_name,
                'feature': feat,
                'test_metrics': test_metrics,
                'params': best_params,
                'history': history
            }

            np.save(os.path.join(results_dir, f"{dataset_name}_{feat}_vgg16_results.npy"), results)
            print(f"✅ {dataset_name}-{feat} TestAcc={test_metrics['accuracy']:.4f}")

# ---------------- Run Pipeline ----------------
if __name__ == "__main__":
    print("🚀 Starting VGG16 Temporal Multi-Dataset Training...")
    improved_grid_search(features, datasets)
    improved_train_and_evaluate(features, datasets)
    print("🎉 All VGG16 training completed!")