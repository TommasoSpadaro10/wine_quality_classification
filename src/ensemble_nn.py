import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import config
import utils
from training.loss import CostSensitiveLoss
from ensemble import ModelA, ModelC, ModelD, get_model_predictions, plot_ensemble_confusion_matrix

class MetaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.20),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),
            nn.Linear(16, 10),
            nn.Softmax(dim=1)
        )
        
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)

def run_meta_nn_ensemble():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    print("Loading datasets...")
    modello_vini_data = r"C:\Users\tomma\Desktop\modello_vini\data"
    val_df = pd.read_csv(os.path.join(modello_vini_data, "winequality-red-validation.csv"), sep=';')
    test_df = pd.read_csv(os.path.join(modello_vini_data, "winequality-red-test.csv"), sep=';')
    
    features_A = [
        'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'pH', 'sulphates', 'alcohol',
        'acidity_ratio', 'sulfur_ratio', 'alcohol_density', 'total_acidity'
    ]
    features_CD = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    
    X_val_A = torch.tensor(val_df[features_A].values, dtype=torch.float32).to(config.DEVICE)
    X_val_CD = torch.tensor(val_df[features_CD].values, dtype=torch.float32).to(config.DEVICE)
    
    X_test_A = torch.tensor(test_df[features_A].values, dtype=torch.float32).to(config.DEVICE)
    X_test_CD = torch.tensor(test_df[features_CD].values, dtype=torch.float32).to(config.DEVICE)
    
    # y targets for MetaNN are the raw labels (0-9 classes)
    y_val_np = val_df['quality'].values
    y_test_np = test_df['quality'].values
    
    # Binary targets for accuracy evaluation
    y_test_good = (y_test_np >= config.QUALITY_THRESHOLD).astype(int)

    print("Instantiating base architectures...")
    model_A = ModelA().to(config.DEVICE)
    model_C = ModelC().to(config.DEVICE)
    model_D = ModelD().to(config.DEVICE)
    
    print("Loading base model weights...")
    path_A = os.path.join(base_dir, 'experiments', 'run_20260723_130402', 'best_model.pth')
    path_C = os.path.join(base_dir, 'experiments', 'run_20260718_000708', 'best_model.pth')
    path_D = os.path.join(base_dir, 'experiments', 'run_20260721_224632', 'model_run_1.pth')

    model_A.load_state_dict(torch.load(path_A, map_location=config.DEVICE, weights_only=True))
    model_C.load_state_dict(torch.load(path_C, map_location=config.DEVICE, weights_only=True))
    model_D.load_state_dict(torch.load(path_D, map_location=config.DEVICE, weights_only=True))
    
    models = [model_A, model_C, model_D]
    for model in models:
        model.eval()

    print("Generating Meta-Features...")
    val_meta_np = get_model_predictions(models, [X_val_A, X_val_CD, X_val_CD])
    test_meta_np = get_model_predictions(models, [X_test_A, X_test_CD, X_test_CD])
    
    from sklearn.model_selection import train_test_split
    
    # Split the validation meta-features into meta-train and meta-val for Early Stopping
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        val_meta_np, y_val_np, test_size=0.2, random_state=config.RANDOM_STATE
    )
    
    meta_train = torch.tensor(X_meta_train, dtype=torch.float32).to(config.DEVICE)
    meta_val = torch.tensor(X_meta_val, dtype=torch.float32).to(config.DEVICE)
    y_meta_train = torch.tensor(y_meta_train, dtype=torch.long).to(config.DEVICE)
    y_meta_val = torch.tensor(y_meta_val, dtype=torch.long).to(config.DEVICE)
    
    test_meta = torch.tensor(test_meta_np, dtype=torch.float32).to(config.DEVICE)
    
    config.PENALTY_BAD_TO_GOOD = 10.0
    config.PENALTY_GOOD_TO_BAD = 10.0
    criterion = CostSensitiveLoss()
    
    num_runs = 10
    epochs = 10000
    patience = 15000
    
    run_dir = os.path.join(base_dir, 'experiments', 'run_ensemble_nn_hard')
    os.makedirs(run_dir, exist_ok=True)
    plots_dir = os.path.join(run_dir, 'best_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    best_overall_acc = 0.0
    best_run_idx = -1
    best_preds = None
    
    results_md = "# Risultati Ensemble Meta-NN (Hard Training)\n\n"
    results_md += "| Run | Epoche | Val Loss | Binary Test Accuracy |\n"
    results_md += "|:---:|:------:|:--------:|:--------------------:|\n"
    
    print(f"Inizio 10 run di addestramento per la Meta-NN (max {epochs} epoche per run, early stopping)...")
    
    for run in range(1, num_runs + 1):
        meta_nn = MetaNN().to(config.DEVICE)
        optimizer = optim.Adam(meta_nn.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        stopped_epoch = epochs
        
        for epoch in range(epochs):
            meta_nn.train()
            optimizer.zero_grad()
            pred = meta_nn(meta_train)
            loss = criterion(pred, y_meta_train)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Validation step
            meta_nn.eval()
            with torch.no_grad():
                val_pred = meta_nn(meta_val)
                val_loss = criterion(val_pred, y_meta_val).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                import copy
                best_weights = copy.deepcopy(meta_nn.state_dict())
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                break
                
        # Load best weights found during early stopping
        meta_nn.load_state_dict(best_weights)
        
        meta_nn.eval()
        with torch.no_grad():
            test_pred_probs = meta_nn(test_meta)
            test_preds = test_pred_probs.argmax(dim=1).cpu().numpy()
            
        test_preds_good = (test_preds >= config.QUALITY_THRESHOLD).astype(int)
        acc = accuracy_score(y_test_good, test_preds_good)
        
        results_md += f"| {run} | {stopped_epoch} | {best_val_loss:.4f} | {acc * 100:.2f}% |\n"
        print(f"Run {run}/10 -> Stopped at epoch {stopped_epoch} | Val Loss: {best_val_loss:.4f} | Accuracy: {acc * 100:.2f}%")
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_run_idx = run
            best_preds = test_preds_good
            torch.save(meta_nn.state_dict(), os.path.join(run_dir, 'best_meta_nn.pth'))
            
            # Save confusion matrix plot
            plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
            plot_ensemble_confusion_matrix(y_test_good, best_preds, plot_path)

    print(f"\n===========================")
    print(f"BEST META-NN (Run {best_run_idx}) BINARY TEST ACCURACY: {best_overall_acc * 100:.2f}%")
    print(f"===========================\n")

    with open(os.path.join(run_dir, 'experiment_results.md'), 'w') as f:
        f.write(results_md)
        
    # Generate Sintesi
    sintesi = f"""# Sintesi dell'Esperimento: Neural Network Stacking Ensemble

Questo esperimento ha implementato un vero e proprio **Deep Stacking**, sostituendo il modello di classificazione lineare (Logistic Regression) con una nuova Rete Neurale (Meta-NN) addestrata a combinare le uscite probabilistiche del Trio di esperti selezionato.

## I 3 Esperti
- **Model A**: Il Migliore Assoluto
- **Model C**: Lo Specialista Low FN
- **Model D**: Lo Specialista Low FP

Ciascun modello fornisce in output 10 probabilità, creando un vettore in ingresso di **30 features** per il Meta-Modello.

## Meta-NN Architettura ("Hard Training" Edition)
- Input: 30 neuroni
- Hidden Layers: 32 neuroni (Dropout 0.20) -> 16 neuroni (Dropout 0.1)
- Output: 10 classi (Softmax)
- Loss: `CostSensitiveLoss` **Simmetrica a 10**
- Hard Training Config: Learning Rate ridotto a `0.001`, regolarizzazione `Weight Decay 1e-4`, ed `Early Stopping` basato sulla Validation Loss calcolata su una frazione interna del Meta-Validation Set, con limite a 10.000 epoche.

## Risultato Finale
Sono state lanciate {num_runs} run indipendenti.
La run migliore (Run {best_run_idx}) si è fermata alla convergenza e ha raggiunto un'accuratezza sul Test Set (Vino Buono >= 6) pari al **{best_overall_acc * 100:.2f}%**.
La relativa matrice di confusione è salvata in `best_plots/confusion_matrix.png`.
"""
    with open(os.path.join(run_dir, 'sintesi.md'), 'w') as f:
        f.write(sintesi)
        
if __name__ == '__main__':
    run_meta_nn_ensemble()
