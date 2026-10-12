import os
import sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import utils
from src.config import config

# --- 1. Define Historical Architectures ---

class ModelA(torch.nn.Module):
    # Best Model (run_20260719_174901 and run_20260723_130402)
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(12, 88),
            torch.nn.BatchNorm1d(88),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(p=0.20),
            torch.nn.Linear(88, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(p=0),
            torch.nn.Linear(10, 10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x): return self.layers(x)

class ModelB(torch.nn.Module):
    # Balanced Model (run_20260716_210922)
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 88),
            torch.nn.BatchNorm1d(88),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.15),
            torch.nn.Linear(88, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(10, 10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x): return self.layers(x)

class ModelC(torch.nn.Module):
    # Asymmetric/Low FP (run_20260718_000708)
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(128, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.10),
            torch.nn.Linear(32, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.0),
            torch.nn.Linear(10, 10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x): return self.layers(x)

class ModelD(torch.nn.Module):
    # Best Model (run_20260719_174901 architecture)
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 88),
            torch.nn.BatchNorm1d(88),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(p=0.20),
            torch.nn.Linear(88, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(p=0),
            torch.nn.Linear(10, 10),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x): return self.layers(x)


def get_model_predictions(models, input_tensor):
    """Passes input through all models and returns concatenated probabilities."""
    all_probs = []
    with torch.no_grad():
        if isinstance(input_tensor, list):
            for model, inp in zip(models, input_tensor):
                probs = model(inp)
                all_probs.append(probs.cpu().numpy())
        else:
            for model in models:
                probs = model(input_tensor)
                all_probs.append(probs.cpu().numpy())
    
    # Concatenate along the feature dimension (axis=1)
    # Output shape: (num_samples, num_models * num_classes) -> (N, 30)
    return np.hstack(all_probs)

def plot_ensemble_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Confusion Matrix (Stacking)')
    plt.xlabel('Predicted Quality')
    plt.ylabel('Actual Quality')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    print("Loading datasets...")
    # Change working directory so utils finds 'data'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    
    val_df = utils.read_csv('winequality-red-validation')
    test_df = utils.read_csv('winequality-red-test')
    
    X_val, y_val = utils.split_df_for_inference(val_df)
    X_test, y_test = utils.split_df_for_inference(test_df)
    
    X_val = X_val.to(config.DEVICE)
    X_test = X_test.to(config.DEVICE)
    y_val_np = y_val.numpy()
    y_test_np = y_test.numpy()

    print("Instantiating architectures...")
    model_A = ModelA().to(config.DEVICE)
    model_B = ModelB().to(config.DEVICE)
    model_C = ModelC().to(config.DEVICE)
    model_D = ModelD().to(config.DEVICE)
    
    print("Loading model weights...")
    # Paths to the selected historical models
    path_A = os.path.join(base_dir, 'experiments', 'run_20260719_174901', 'model_run_10.pth')
    path_B = os.path.join(base_dir, 'experiments', 'run_20260716_210922', 'best_model.pth')
    path_C = os.path.join(base_dir, 'experiments', 'run_20260718_000708', 'best_model.pth')
    path_D = os.path.join(base_dir, 'experiments', 'run_20260721_224632', 'model_run_1.pth')

    print("Loading model weights...")
    model_A.load_state_dict(torch.load(path_A, map_location=config.DEVICE, weights_only=True))
    model_B.load_state_dict(torch.load(path_B, map_location=config.DEVICE, weights_only=True))
    model_C.load_state_dict(torch.load(path_C, map_location=config.DEVICE, weights_only=True))
    model_D.load_state_dict(torch.load(path_D, map_location=config.DEVICE, weights_only=True))
    
    models = [model_A, model_B, model_C, model_D]
    for model in models:
        model.eval()
        
    y_val_good = (y_val_np >= config.QUALITY_THRESHOLD).astype(int)
    y_test_good = (y_test_np >= config.QUALITY_THRESHOLD).astype(int)

    print("Generating base model predictions for Validation Set (for clean Meta-Model training)...")
    val_meta_features = get_model_predictions(models, X_val)
    
    print("Training Logistic Regression Meta-Model...")
    # Meta-model initialization
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(val_meta_features, y_val_good)
    
    print("Generating base model predictions for Test Set...")
    test_meta_features = get_model_predictions(models, X_test)
    
    # Calculate base model accuracies
    model_names = ["Model A (Best)", "Model B (Balanced)", "Model C (Low FN)", "Model D (Low FP)"]
    for i, model_name in enumerate(model_names):
        preds = np.argmax(test_meta_features[:, i*10:(i+1)*10], axis=1)
        preds_good = (preds >= config.QUALITY_THRESHOLD).astype(int)
        acc = accuracy_score(y_test_good, preds_good)
        print(f"{model_name} Binary Test Accuracy: {acc * 100:.2f}%")
        
    print("Predicting final ensemble classes...")
    y_pred_ensemble = meta_model.predict(test_meta_features)
    
    acc = accuracy_score(y_test_good, y_pred_ensemble)
    print(f"\n===========================")
    print(f"ENSEMBLE BINARY TEST ACCURACY: {acc * 100:.2f}%")
    print(f"===========================\n")
    
    plot_path = os.path.join(base_dir, 'plots', 'ensemble_confusion_matrix.png')
    plot_ensemble_confusion_matrix(y_test_good, y_pred_ensemble, plot_path)
    print(f"Confusion matrix saved to {plot_path}")

if __name__ == '__main__':
    main()
