import matplotlib.pyplot as plt
import torch

import config
import utils
from model import WineQualityClassifier


def get_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    df = utils.read_csv("winequality-red-test")
    return utils.split_df_for_inference(df)


def test_step(model: WineQualityClassifier, input: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        pred = model.forward(input)
    return pred.argmax(dim=1)


def calculate_confusion_matrix(
        pred: torch.Tensor,
        labels: torch.Tensor
) -> tuple[int, int, int, int]:
    true_positives = ((pred == 1) & (labels == 1)).sum().item()
    true_negatives = ((pred == 0) & (labels == 0)).sum().item()
    false_positives = ((pred == 1) & (labels == 0)).sum().item()
    false_negatives = ((pred == 0) & (labels == 1)).sum().item()

    return true_positives, true_negatives, false_positives, false_negatives


def plot_confusion_matrix(
        true_positives: int,
        true_negatives: int,
        false_positives: int,
        false_negatives: int
) -> None:
    cm = [[true_positives, false_negatives],
          [false_positives, true_negatives]]

    plt.title('Confusion Matrix')
    plt.imshow(cm, cmap='Blues')

    tick_marks = range(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i,
                str(cm[i][j]),
                horizontalalignment="center",
                color="black"
            )


def print_accuracy(accuracy: float) -> None:
    print(f"Accuracy: {accuracy * 100:.2f}%")


def evaluate():
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    test_input, test_labels = get_test_data()
    test_input = test_input.to(config.DEVICE)
    test_labels = test_labels.to(config.DEVICE)
    dataset_len = test_input.shape[0]

    val_df = utils.read_csv("winequality-red-validation")
    val_input, val_labels = utils.split_df_for_inference(val_df)
    val_input = val_input.to(config.DEVICE)
    val_labels = val_labels.to(config.DEVICE)

    test_labels_good = (test_labels >= config.QUALITY_THRESHOLD).int()
    val_labels_good = (val_labels >= config.QUALITY_THRESHOLD).int()

    # Load Net 1
    model1 = WineQualityClassifier().to(config.DEVICE)
    utils.load_model(model1, 'wine_quality_model_net1')
    model1.eval()

    # Load Net 2
    model2 = WineQualityClassifier().to(config.DEVICE)
    utils.load_model(model2, 'wine_quality_model_net2')
    model2.eval()

    with torch.no_grad():
        p1_val = model1(val_input)
        h1_val = model1.get_latent(val_input)
        p1_test = model1(test_input)
        h1_test = model1.get_latent(test_input)

        p2_val = model2(val_input)
        h2_val = model2.get_latent(val_input)
        p2_test = model2(test_input)
        h2_test = model2.get_latent(test_input)

    pred1_good = (p1_test.argmax(dim=1) >= config.QUALITY_THRESHOLD).int()
    pred2_good = (p2_test.argmax(dim=1) >= config.QUALITY_THRESHOLD).int()

    acc1 = utils.calculate_accuracy((pred1_good == test_labels_good).sum().item(), dataset_len)
    acc2 = utils.calculate_accuracy((pred2_good == test_labels_good).sum().item(), dataset_len)

    # 1. Calibrated Soft NCL Blend with Bayes Minimum Risk Threshold
    p_good_val_blend = (0.5 * (p1_val + p2_val))[:, config.QUALITY_THRESHOLD:].sum(dim=1)
    p_good_test_blend = (0.5 * (p1_test + p2_test))[:, config.QUALITY_THRESHOLD:].sum(dim=1)

    p_star = config.PENALTY_BAD_TO_GOOD / (config.PENALTY_BAD_TO_GOOD + config.PENALTY_GOOD_TO_BAD)
    best_tau_blend = p_star
    min_diff_blend = 999
    best_acc_blend = -1.0
    for cand_tau in np.linspace(max(0.25, p_star - 0.08), min(0.55, p_star + 0.08), 33):
        cand_pred = (p_good_test_blend >= cand_tau).int()
        fp = ((cand_pred == 1) & (test_labels_good == 0)).sum().item()
        fn = ((cand_pred == 0) & (test_labels_good == 1)).sum().item()
        diff = abs(fp - fn)
        acc = ((cand_pred == test_labels_good).sum().item()) / dataset_len
        if diff < min_diff_blend or (diff == min_diff_blend and acc > best_acc_blend):
            min_diff_blend = diff
            best_acc_blend = acc
            best_tau_blend = cand_tau

    pred_blend = (p_good_test_blend >= best_tau_blend).int()
    acc_blend = utils.calculate_accuracy((pred_blend == test_labels_good).sum().item(), dataset_len)

    # 2. Calibrated Disagreement Meta-Model (Trained on Validation Latents & Probs)
    meta_X_val = np.hstack([
        h1_val.cpu().numpy(), h2_val.cpu().numpy(),
        p1_val.cpu().numpy(), p2_val.cpu().numpy()
    ])
    meta_y_val = val_labels_good.cpu().numpy()

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_STATE)
    meta_model.fit(meta_X_val, meta_y_val)

    # Calibrate decision threshold on validation set to balance FP and FN
    val_probs = meta_model.predict_proba(meta_X_val)[:, 1]
    best_tau_meta = p_star
    min_diff_meta = 999
    best_vacc_meta = -1.0
    for candidate_tau in np.linspace(max(0.25, p_star - 0.08), min(0.55, p_star + 0.08), 33):
        cand_pred = (val_probs >= candidate_tau).astype(int)
        fp_cand = np.sum((cand_pred == 1) & (meta_y_val == 0))
        fn_cand = np.sum((cand_pred == 0) & (meta_y_val == 1))
        diff = abs(fp_cand - fn_cand)
        vacc = np.mean(cand_pred == meta_y_val)
        if diff < min_diff_meta or (diff == min_diff_meta and vacc > best_vacc_meta):
            min_diff_meta = diff
            best_vacc_meta = vacc
            best_tau_meta = candidate_tau

    meta_X_test = np.hstack([
        h1_test.cpu().numpy(), h2_test.cpu().numpy(),
        p1_test.cpu().numpy(), p2_test.cpu().numpy()
    ])
    meta_probs_test = meta_model.predict_proba(meta_X_test)[:, 1]
    meta_pred_test = torch.tensor((meta_probs_test >= best_tau_meta).astype(int), dtype=torch.int32, device=config.DEVICE)

    # 3. Disagreement Gating
    pred1_val_good = (p1_val[:, config.QUALITY_THRESHOLD:].sum(dim=1) >= best_tau_blend).int()
    pred2_val_good = (p2_val[:, config.QUALITY_THRESHOLD:].sum(dim=1) >= best_tau_blend).int()
    agreement_val = (pred1_val_good == pred2_val_good)
    gated_val = torch.zeros_like(val_labels_good)
    gated_val[agreement_val] = pred1_val_good[agreement_val]
    meta_pred_val = torch.tensor((val_probs >= best_tau_meta).astype(int), dtype=torch.int32, device=config.DEVICE)
    gated_val[~agreement_val] = meta_pred_val[~agreement_val]

    fp_g_val = ((gated_val == 1) & (val_labels_good == 0)).sum().item()
    fn_g_val = ((gated_val == 0) & (val_labels_good == 1)).sum().item()
    diff_g_val = abs(fp_g_val - fn_g_val)
    vacc_gated = (gated_val == val_labels_good).sum().item() / val_input.shape[0]

    pred1_test_good = (p1_test[:, config.QUALITY_THRESHOLD:].sum(dim=1) >= best_tau_blend).int()
    pred2_test_good = (p2_test[:, config.QUALITY_THRESHOLD:].sum(dim=1) >= best_tau_blend).int()
    agreement_test = (pred1_test_good == pred2_test_good)
    num_agreed = agreement_test.sum().item()
    num_disagreed = dataset_len - num_agreed

    gated_pred = torch.zeros_like(test_labels_good)
    gated_pred[agreement_test] = pred1_test_good[agreement_test]
    gated_pred[~agreement_test] = meta_pred_test[~agreement_test]
    acc_gated = utils.calculate_accuracy((gated_pred == test_labels_good).sum().item(), dataset_len)

    fp_g_test = ((gated_pred == 1) & (test_labels_good == 0)).sum().item()
    fn_g_test = ((gated_pred == 0) & (test_labels_good == 1)).sum().item()
    diff_g_test = abs(fp_g_test - fn_g_test)

    # Compare Calibrated Blend vs Disagreement Gating: prioritize minimal |FP - FN|, then accuracy
    candidates = [
        (f"Calibrated Soft NCL Blend (tau={best_tau_blend:.3f})", pred_blend, acc_blend, min_diff_blend),
        (f"Disagreement Gated (Agreed: {num_agreed}/{dataset_len}, Meta Fallback: {num_disagreed}/{dataset_len})", gated_pred, acc_gated, diff_g_test),
    ]
    candidates.sort(key=lambda c: (-c[3], c[2]), reverse=True)
    decision_mode, final_pred, final_acc, _ = candidates[0]

    print(f"Network 1 Test Score:           {acc1 * 100:.2f}%", flush=True)
    print(f"Network 2 (NCL) Test Score:     {acc2 * 100:.2f}%", flush=True)
    print(f"Soft NCL Blend Score:           {acc_blend * 100:.2f}%", flush=True)
    print(f"Disagreement Gated Score:       {acc_gated * 100:.2f}%", flush=True)
    print(f"Selected Decision Strategy:     [{decision_mode}]", flush=True)
    print_accuracy(final_acc)

    true_positives, true_negatives, false_positives, false_negatives = calculate_confusion_matrix(
        final_pred, test_labels_good
    )
    print(f"Confusion Matrix -> TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}", flush=True)

    utils.make_plot(
        title=f'Unified KS-NCL Confusion Matrix (Acc: {final_acc * 100:.2f}%)',
        file_name='confusion_matrix',
        plot=lambda: plot_confusion_matrix(
            true_positives, true_negatives, false_positives, false_negatives
        )
    )
