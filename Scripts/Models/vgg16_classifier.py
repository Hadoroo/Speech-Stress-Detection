import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from matplotlib.pylab import f
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from Scripts.Data_Collector.temporal_feature_collector import TemporalFeatures

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------------- Feature Type ----------------
features = ["MFCC", "GFCC", "LogFBank", "f0"]

# ---------------- Improved Collate Function ----------------
def collate_fn(batch):
    feats, labels, fnames = zip(*batch)
    feats = torch.stack(feats)
    labels = torch.tensor(labels, dtype=torch.long)
    return feats, labels, fnames

# ---------------- Improved VGG16 Model (Fixed BatchNorm Issue) ----------------
class ImprovedVGG16Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout_rate=0.5, hidden_dim=512):
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
        
        # Use LayerNorm instead of BatchNorm for stability with small batch sizes
        self.layer_norm = nn.LayerNorm(hidden_dim // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(hidden_dim // 8, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_processor(x)
        x = self.layer_norm(x)  # LayerNorm works with any batch size
        x = self.classifier(x)
        return x

# ---------------- Calculate Class Weights ----------------
def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced datasets"""
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label)
    
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Inverse frequency weighting
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ---------------- Safe Batch Training Function ----------------
def safe_batch_training(model, train_loader, criterion, optimizer, device):
    """Handle training with potential small batch sizes"""
    model.train()
    train_loss = 0
    batch_count = 0
    
    for X, y, _ in train_loader:
        # Skip batches with only 1 sample
        if X.size(0) == 1:
            continue
            
        X, y = X.to(device), y.to(device)
        X = X.view(X.size(0), -1).float()
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        batch_count += 1
    
    return train_loss / max(batch_count, 1)  # Avoid division by zero

# ---------------- Improved Grid Search with Safe Training ----------------
def improved_grid_search(features):
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [16, 32],
        'hidden_dim': [256, 512],
        'dropout_rate': [0.3, 0.5],
        'weight_decay': [1e-5, 0.0]
    }
    
    save_dir = "Results/Tuning/VGG16_Classifier"
    os.makedirs(save_dir, exist_ok=True)

    for feat in features:
        save_path = os.path.join(save_dir, f"{feat}_improved.npy")
        if os.path.exists(save_path):
            print(f"⏩ Skip {feat}, improved tuning exists: {save_path}")
            continue

        print(f"\n🔎 Improved grid search untuk fitur: {feat}")

        train_ds = TemporalFeatures(model_state="train", acoustic_feature_type=feat, temporal_feature_type="summary")
        test_ds  = TemporalFeatures(model_state="test",  acoustic_feature_type=feat, temporal_feature_type="summary")

        sample_feat, _, _ = train_ds[0]
        input_dim = sample_feat.numel()

        best_acc = 0.0
        best_params = None
        
        param_list = list(ParameterGrid(param_grid))
        
        for params in tqdm(param_list, desc=f"Tuning {feat}"):
            lr = params['learning_rate']
            bs = params['batch_size']
            hidden_dim = params['hidden_dim']
            dropout_rate = params['dropout_rate']
            weight_decay = params['weight_decay']
            
            actual_bs = min(bs, len(train_ds))
            if actual_bs < 2:
                continue
                
            train_loader = DataLoader(train_ds, batch_size=actual_bs, shuffle=True, 
                                    collate_fn=collate_fn, num_workers=2, pin_memory=True)
            test_loader  = DataLoader(test_ds, batch_size=actual_bs, shuffle=False, 
                                    collate_fn=collate_fn, num_workers=2, pin_memory=True)

            model = ImprovedVGG16Classifier(
                input_dim=input_dim, 
                num_classes=3, 
                dropout_rate=dropout_rate,
                hidden_dim=hidden_dim
            ).to(device)
            
            class_weights = calculate_class_weights(train_ds)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

            quick_epochs = 3
            current_best_val_acc = 0
            
            for epoch in range(quick_epochs):
                train_loss = safe_batch_training(model, train_loader, criterion, optimizer, device)
                scheduler.step()
                
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X, y, _ in test_loader:
                        if X.size(0) == 1:
                            continue
                        X, y = X.to(device), y.to(device)
                        X = X.view(X.size(0), -1).float()
                        outputs = model(X)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += y.size(0)
                        val_correct += (predicted == y).sum().item()
                
                if val_total > 0:
                    val_acc = val_correct / val_total
                    current_best_val_acc = max(current_best_val_acc, val_acc)
                
                if epoch >= 1 and val_total > 0 and val_acc < current_best_val_acc * 0.8:
                    break

            if val_total > 0 and current_best_val_acc > best_acc:
                best_acc = current_best_val_acc
                best_params = params

        if best_params is not None:
            result = {
                'params': best_params,
                'best_acc': best_acc,
                'input_dim': input_dim
            }
            np.save(save_path, result)
            print(f"✅ Best params for {feat}: {best_params}")
            print(f"✅ Best validation accuracy: {best_acc:.4f}")
        else:
            print(f"❌ No valid parameters found for {feat}")

# ---------------- Safe Evaluation Function ----------------
def safe_evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y, _ in dataloader:
            if X.size(0) == 1:
                continue
            X = X.to(device)
            X = X.view(X.size(0), -1).float()
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted.cpu() == y).sum().item()
    
    return correct / total if total > 0 else 0.0

# ---------------- Improved Training with Safe Batch Handling ----------------
def improved_train_and_evaluate(features):
    model_dir = "Results/Models/VGG16_Improved"
    os.makedirs(model_dir, exist_ok=True)
    results_dir = "Results/Predictions/VGG16_Improved"
    os.makedirs(results_dir, exist_ok=True)

    for feat in features:
        tune_path = f"Results/Tuning/VGG16_Classifier/{feat}_improved.npy"
        if not os.path.exists(tune_path):
            print(f"Skip {feat}, improved tuning file not found: {tune_path}")
            continue
        
        tune_result = np.load(tune_path, allow_pickle=True).item()
        best_params = tune_result['params']
        input_dim = tune_result['input_dim']
        
        lr = best_params['learning_rate']
        bs = best_params['batch_size']
        hidden_dim = best_params['hidden_dim']
        dropout_rate = best_params['dropout_rate']
        weight_decay = best_params['weight_decay']
        
        print(f"\n🚀 Training {feat} with best params: {best_params}")

        train_ds = TemporalFeatures(model_state="train", acoustic_feature_type=feat, temporal_feature_type="summary")
        test_ds  = TemporalFeatures(model_state="test",  acoustic_feature_type=feat, temporal_feature_type="summary")

        actual_bs = min(bs, len(train_ds) - 1)
        if actual_bs < 2:
            actual_bs = 16
            
        print(f"Using batch size: {actual_bs}")

        train_loader = DataLoader(train_ds, batch_size=actual_bs, shuffle=True, 
                                collate_fn=collate_fn, drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=actual_bs, shuffle=False, 
                                collate_fn=collate_fn, drop_last=False)

        model = ImprovedVGG16Classifier(
            input_dim=input_dim,
            num_classes=3,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim
        ).to(device)
        
        class_weights = calculate_class_weights(train_ds)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X, y, _ in train_loader:
                X, y = X.to(device), y.to(device)
                X = X.view(X.size(0), -1).float()
                
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_acc'].append(train_acc)
            
            val_acc = safe_evaluate_model(model, test_loader, device)
            training_history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(model_dir, f"{feat}_best_vgg16.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'params': best_params
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        checkpoint_path = os.path.join(model_dir, f"{feat}_best_vgg16.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Best validation accuracy for {feat}: {checkpoint['val_acc']:.4f}")

        def comprehensive_evaluation(dataloader, split_name):
            model.eval()
            all_preds = []
            all_trues = []
            all_probs = []
            all_fnames = []
            
            with torch.no_grad():
                for X, y, fnames in dataloader:
                    if X.size(0) == 1:
                        continue
                    X = X.to(device)
                    X = X.view(X.size(0), -1).float()
                    outputs = model(X)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_trues.extend(y.numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_fnames.extend(fnames)
            
            if len(all_preds) > 0:
                accuracy = (np.array(all_preds) == np.array(all_trues)).mean()
            else:
                accuracy = 0.0
            
            return {
                'preds': np.array(all_preds),
                'trues': np.array(all_trues),
                'probs': np.array(all_probs),
                'fnames': np.array(all_fnames),
                'accuracy': accuracy
            }

        train_results = comprehensive_evaluation(train_loader, "train")
        test_results = comprehensive_evaluation(test_loader, "test")
        
        results = {
            'train': train_results,
            'test': test_results,
            'training_history': training_history,
            'best_params': best_params,
            'best_val_acc': best_val_acc
        }
        
        np.save(os.path.join(results_dir, f"{feat}_comprehensive_results.npy"), results)
        
        print(f"\n📊 Final Results for {feat}:")
        print(f"Train Accuracy: {train_results['accuracy']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"✅ Finished improved training & evaluation for {feat}")

# ---------------- Run Improved Pipeline ----------------
if __name__ == "__main__":
    print("🚀 Starting Improved Training Pipeline...")
    
    improved_grid_search(features)
    improved_train_and_evaluate(features)
    
    print("🎉 Improved pipeline completed!")
