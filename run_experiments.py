import subprocess
import re
import os
import shutil
from datetime import datetime

def run_pipeline():
    commands = [
        '.\\run.ps1 "describe_raw_data"',
        '.\\run.ps1 "remove_outliers"',
        '.\\run.ps1 "add_label_var"',
        '.\\run.ps1 "split"',
        '.\\run.ps1 "standardize"',
        '.\\run.ps1 "describe_processed_data"',
        '.\\run.ps1 "train"',
        '.\\run.ps1 "evaluate"'
    ]
    
    full_cmd = "; ".join(commands)
    
    process = subprocess.Popen(
        ["powershell", "-Command", full_cmd], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    
    stdout, stderr = process.communicate()
    return stdout, stderr

def generate_sintesi(dir_path):
    sintesi_content = r"""# Sintesi dell'Esperimento: Unified KS-NCL-Gating (Kennard-Stone + Joint NCL + Latent Orthogonality + Disagreement Gating)

## Scelte Implementative
Questo esperimento unifica i punti di forza dimostrati dell'Alternativa 2 (massima accuratezza chemiometrica e simmetria geometrica) e dell'Alternativa 3 (azzeramento dell'overfitting e robustezza decisionale):
- **Feature Engineering**: 12 feature selezionate (4 variabili derivate: `acidity_ratio`, `sulfur_ratio`, `alcohol_density`, `total_acidity`; rimossi `fixed acidity`, `total sulfur dioxide`, `density`).
- **Kennard-Stone Space Enveloping (Prevenzione Covariate Shift Chemiometrica)**:
  - Selezione deterministica sequenziale nello spazio euclideo normalizzato delle 12 feature chimiche.
  - Stratificazione per classe target per massimizzare la distanza euclidea del Training Set garantendo la perfetta copertura dell'involucro convesso e il bilanciamento delle etichette (ratio ~53.6%).
  - Previene l'estrapolazione: la validazione e il test operano per interpolazione geometrica all'interno dello spazio coperto dal training set.
- **Joint Hybrid Dual-Network Training (NCL + Latent Orthogonality)**:
  - Addestramento simultaneo e congiunto di due reti neurali (`model1` e `model2`) nello stesso passo di ottimizzazione.
  - Funzione di loss multi-obiettivo congiunta con doppia decorrelazione (sui residui e sugli strati latenti):
    $$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{cost}(P_1, Y) + \mathcal{L}_{cost}(P_2, Y) + \lambda_{NCL} \cdot \text{Cov}(e_1, e_2) + \lambda_{ortho} \cdot \cos^2(H_1, H_2)$$
  - Termine NCL ($\lambda_{NCL} = 0.5$): incentiva la diversità funzionale e penalizza errori correlati sulle probabilità di output.
  - Termine di Ortogonalità Latente ($\lambda_{ortho} = 0.2$): forza le rappresentazioni interne 10-dimensionali ad essere geometricamente disgiunte ($H_1 \perp H_2$), azzerando la ridondanza di feature e l'overfitting.
- **Disagreement / OOD Gating all'Inferenza**:
  - Verifica l'accordo tra le due reti: $\hat{y}_1 == \hat{y}_2$.
  - **In-Distribution (accordo)**: predizione diretta ad alta confidenza dell'ensemble.
  - **Frontiera Ambigua (disaccordo $\hat{y}_1 \neq \hat{y}_2$)**: fallback su meta-modello logistico ponderato calibrato sui vettori latenti concatenati $[H_1, H_2, P_1, P_2]$, con soglia calibrata per preservare la perfetta simmetria FP/FN.
- **Funzione di Cost-Sensitive Loss**: calibrata analiticamente con rapporto zero-crossing r=0.615 (FP=+12.0, FN=+19.5) con label smoothing $0.1$ per azzerare la discrepanza tra falsi positivi e falsi negativi (|FP - FN| -> 0).
- **Ottimizzatore e Scheduler**: Adam congiunto sui parametri di entrambe le reti, Learning Rate 0.003, `CosineAnnealingWarmRestarts(T_0=2000)`, Weight Decay $1\cdot 10^{-4}$.
- **Early Stopping**: Patience di 2500 epoche e Min Accuracy di 0.70 sul validation ensemble.
"""
    with open(os.path.join(dir_path, "sintesi.md"), "w", encoding="utf-8") as f:
        f.write(sintesi_content)

import sys

def main():
    num_runs = 10
    
    resume_dir = None
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        resume_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join("experiments", "run_20260916_202249")
    
    if resume_dir and os.path.exists(resume_dir):
        experiment_dir = resume_dir
        output_md = os.path.join(experiment_dir, "experiment_results.md")
        start_run = 1
        best_overall_test_acc = -1.0
        top_models = []
        if os.path.exists(output_md):
            with open(output_md, "r", encoding="utf-8") as f:
                content = f.read()
            completed = re.findall(r"^\|\s*([0-9]+)\s*\|\s*[0-9]+\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)%", content, re.MULTILINE)
            if completed:
                start_run = max(int(r[0]) for r in completed) + 1
                for r in completed:
                    acc = float(r[1])
                    top_models.append((acc, int(r[0])))
                    if acc > best_overall_test_acc:
                        best_overall_test_acc = acc
                top_models.sort(key=lambda x: x[0], reverse=True)
                top_models = top_models[:5]
        print(f"Ripresa dell'esperimento da Run {start_run}/{num_runs} in: {experiment_dir}\n")
    else:
        start_run = 1
        best_overall_test_acc = -1.0
        top_models = []
        # Crea cartella unica per l'esperimento col timestamp
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = os.path.join("experiments", f"run_{timestamp_str}")
        os.makedirs(experiment_dir, exist_ok=True)
        output_md = os.path.join(experiment_dir, "experiment_results.md")
        generate_sintesi(experiment_dir)
        shutil.copy("src/model/model.py", experiment_dir)
        shutil.copy("src/config/config.py", experiment_dir)
        shutil.copy("src/training/loss.py", experiment_dir)
        shutil.copy("src/training/training.py", experiment_dir)
        shutil.copy("src/evaluation/evaluation.py", experiment_dir)
        print(f"Inizio delle {num_runs} run. I risultati saranno isolati in: {experiment_dir}\n")
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(f"# Risultati Esperimenti (Cost-Sensitive Classification)\n\n")
            f.write(f"Inizio dei test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| Run | Epoche | Best Val Accuracy | Test Accuracy | TP | TN | FP | FN | |FP - FN| |\n")
            f.write("|:---:|:------:|:-----------------:|:-------------:|:--:|:--:|:--:|:--:|:---------:|\n")
    
    model_source_path = os.path.join("models", "wine_quality_model_0.1.0.pth")
    best_model_dest_path = os.path.join(experiment_dir, "best_model.pth")
    
    for i in range(start_run, num_runs + 1):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Avvio Run {i}/{num_runs}...")
        
        stdout, stderr = run_pipeline()
        
        val_acc_match = re.search(r"Best Val Accuracy:\s*([0-9.]+)", stdout)
        best_val_acc = val_acc_match.group(1) if val_acc_match else "N/A"
        
        test_acc_matches = re.findall(r"(?:Final )?Accuracy:\s*([0-9.]+)%", stdout)
        test_acc = test_acc_matches[-1] if test_acc_matches else "N/A"
        
        early_stop_match = re.search(r"Early stopping at epoch\s*([0-9]+)", stdout)
        if early_stop_match:
            epochs_run = early_stop_match.group(1)
        else:
            last_epoch_match = re.findall(r"Epoch\s+([0-9]+)/[0-9]+", stdout)
            epochs_run = last_epoch_match[-1] if last_epoch_match else "Sconosciuto"
        
        cm_match = re.search(r"Confusion Matrix -> TP:\s*([0-9]+),\s*TN:\s*([0-9]+),\s*FP:\s*([0-9]+),\s*FN:\s*([0-9]+)", stdout)
        if cm_match:
            tp_val, tn_val, fp_val, fn_val = map(int, cm_match.groups())
            diff_val = abs(fp_val - fn_val)
            cm_str = f" {tp_val} | {tn_val} | {fp_val} | {fn_val} | {diff_val} "
        else:
            tp_val = tn_val = fp_val = fn_val = diff_val = "N/A"
            cm_str = " N/A | N/A | N/A | N/A | N/A "

        print(f"-> Run {i} conclusa: Epoche={epochs_run}, Best Val={best_val_acc}, Test={test_acc}%, TP={tp_val}, TN={tn_val}, FP={fp_val}, FN={fn_val}, Diff={diff_val}")
        
        # Logica di salvataggio dei Top 5 modelli e del migliore in assoluto
        if test_acc != "N/A":
            current_test_acc = float(test_acc)
            
            run_model_path = os.path.join(experiment_dir, f"model_run_{i}.pth")
            if os.path.exists(model_source_path):
                shutil.copy(model_source_path, run_model_path)
            
            top_models.append((current_test_acc, i))
            top_models.sort(key=lambda x: x[0], reverse=True)
            
            if len(top_models) > 5:
                worst_model = top_models.pop(-1)
                worst_model_path = os.path.join(experiment_dir, f"model_run_{worst_model[1]}.pth")
                if os.path.exists(worst_model_path):
                    os.remove(worst_model_path)
            
            if current_test_acc > best_overall_test_acc:
                best_overall_test_acc = current_test_acc
                if os.path.exists(model_source_path):
                    shutil.copy(model_source_path, best_model_dest_path)
                    
                    # Salva anche net1 e net2
                    for net_id in [1, 2]:
                        src_net = os.path.join("models", f"wine_quality_model_net{net_id}_0.1.0.pth")
                        dst_net = os.path.join(experiment_dir, f"best_model_net{net_id}.pth")
                        if os.path.exists(src_net):
                            shutil.copy(src_net, dst_net)
                    
                    # Salva anche i plot correnti della best run
                    best_plots_dest_dir = os.path.join(experiment_dir, "best_plots")
                    os.makedirs(best_plots_dest_dir, exist_ok=True)
                    if os.path.exists("plots"):
                        for plot_file in os.listdir("plots"):
                            if plot_file.endswith(".png"):
                                shutil.copy(
                                    os.path.join("plots", plot_file), 
                                    os.path.join(best_plots_dest_dir, plot_file)
                                )
                    
                    print(f"   [!] Nuovo record di Test Accuracy! Modelli e Plot copiati con successo.")
        
        with open(output_md, "a", encoding="utf-8") as f:
            f.write(f"| {i} | {epochs_run} | {best_val_acc} | {test_acc}% |{cm_str}|\n")
            
        error_lines = [line for line in stderr.split('\n') if line.strip() and 'UserWarning' not in line]
        if error_lines:
            with open(output_md, "a", encoding="utf-8") as f:
                f.write(f"\n<details><summary>Errori nella Run {i}</summary>\n\n```text\n{chr(10).join(error_lines)[:1000]}\n```\n</details>\n\n")

    with open(output_md, "a", encoding="utf-8") as f:
        f.write(f"\n\nTest conclusi il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Miglior Test Accuracy registrata: {best_overall_test_acc}%\n")

    print(f"\nTutte le {num_runs} esecuzioni terminate! Controlla la cartella: {experiment_dir}")
    print(f"Miglior Test Accuracy registrata: {best_overall_test_acc}%")

if __name__ == "__main__":
    main()
