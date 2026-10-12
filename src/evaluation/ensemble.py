import os
import glob
import torch
import sys

# Aggiunge src al path per poter importare i moduli correttamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import utils
from model import WineQualityClassifier
from evaluation import get_test_data, calculate_confusion_matrix, plot_confusion_matrix, print_accuracy

def run_ensemble(experiment_dir: str):
    print(f"Inizializzazione Ensemble da {experiment_dir}...")
    model_paths = glob.glob(os.path.join(experiment_dir, "model_run_*.pth"))
    
    if not model_paths:
        print("Nessun modello trovato in questa cartella.")
        return
        
    print(f"Trovati {len(model_paths)} modelli. Caricamento in corso...")
    
    models = []
    for path in model_paths:
        model = WineQualityClassifier()
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.to(config.DEVICE)
        model.eval()
        models.append(model)
        
    test_input, test_labels = get_test_data()
    test_input = test_input.to(config.DEVICE)
    test_labels = test_labels.to(config.DEVICE)
    
    dataset_len = test_input.shape[0]
    
    # Preallocazione del tensore per accumulare le probabilità
    ensemble_probs = torch.zeros((dataset_len, config.NUM_CLASSES), device=config.DEVICE)
    
    print(f"Esecuzione dell'inferenza combinata (Softmax Averaging)...")
    with torch.no_grad():
        for model in models:
            probs = model.forward(test_input)
            ensemble_probs += probs
            
    # Media aritmetica delle probabilità
    ensemble_probs /= len(models)
    
    # Estrazione classe con probabilità massima
    pred = ensemble_probs.argmax(dim=1)
    
    pred_good = (pred >= config.QUALITY_THRESHOLD).int()
    test_labels_good = (test_labels >= config.QUALITY_THRESHOLD).int()
    
    num_correct = (pred_good == test_labels_good).sum().item()
    accuracy = utils.calculate_accuracy(num_correct, dataset_len)
    
    print("\n" + "="*30)
    print("      RISULTATI ENSEMBLE")
    print("="*30)
    print_accuracy(accuracy)
    
    true_positives, true_negatives, false_positives, false_negatives = calculate_confusion_matrix(
        pred_good, test_labels_good
    )
    
    print(f"True Positives:  {true_positives}")
    print(f"True Negatives:  {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    
    # Salva il plot della matrice di confusione
    utils.make_plot(
        title=f'Ensemble Confusion Matrix (N={len(models)})',
        file_name='ensemble_confusion_matrix',
        plot=lambda: plot_confusion_matrix(
            true_positives, true_negatives, false_positives, false_negatives
        )
    )
    print("\nLa matrice di confusione dell'ensemble è stata salvata in 'plots/ensemble_confusion_matrix.png'")

if __name__ == "__main__":
    # Se non viene passato un argomento, cerca l'ultima cartella creata in 'experiments'
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments'))
        runs = glob.glob(os.path.join(base_dir, "run_*"))
        if not runs:
            print("Nessuna cartella di esperimenti trovata.")
            sys.exit(1)
        runs.sort()
        exp_dir = runs[-1]
        
    run_ensemble(exp_dir)
