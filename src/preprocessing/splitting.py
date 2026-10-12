import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import config
import utils


def kennard_stone_indices(X: np.ndarray, k: int) -> np.ndarray:
    """
    Kennard-Stone algorithm:
    Selects k samples that uniformly cover the Euclidean space and envelope the convex hull.
    """
    n = len(X)
    if k >= n:
        return np.arange(n)

    # Compute pairwise Euclidean distance matrix
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

    # Find the two points with the maximum pairwise distance
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [int(i), int(j)]
    remaining = [idx for idx in range(n) if idx not in selected]

    # Minimum distance of remaining points to any selected point
    min_dists = np.min(dist[remaining][:, selected], axis=1)

    while len(selected) < k:
        next_pos = np.argmax(min_dists)
        chosen = remaining[next_pos]
        selected.append(chosen)
        remaining.pop(next_pos)
        min_dists = np.delete(min_dists, next_pos)

        if len(remaining) > 0:
            new_dists = dist[remaining, chosen]
            min_dists = np.minimum(min_dists, new_dists)

    return np.array(selected, dtype=int)


def stratified_kennard_stone(
    X: np.ndarray,
    y_labels: np.ndarray,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VALIDATION_RATIO
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Class-Stratified Kennard-Stone:
    Applies Kennard-Stone within each target class to guarantee both
    maximal Euclidean space enveloping and exact class balance.
    """
    train_idx = []
    val_idx = []
    test_idx = []

    for cls in np.unique(y_labels):
        cls_indices = np.where(y_labels == cls)[0]
        X_cls = X[cls_indices]
        n_cls = len(cls_indices)
        n_tr = int(round(train_ratio * n_cls))
        n_va = int(round(val_ratio * n_cls))

        # 1. Select Train set using Kennard-Stone on class samples
        tr_sel = kennard_stone_indices(X_cls, n_tr)
        train_idx.extend(cls_indices[tr_sel])

        # 2. Select Validation set using Kennard-Stone on remaining samples
        rem_cls_indices = np.delete(cls_indices, tr_sel)
        X_rem = X[rem_cls_indices]
        va_sel = kennard_stone_indices(X_rem, n_va)
        val_idx.extend(rem_cls_indices[va_sel])

        # 3. Remainder forms Test set
        te_sel = np.delete(np.arange(len(rem_cls_indices)), va_sel)
        test_idx.extend(rem_cls_indices[te_sel])

    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int), np.array(test_idx, dtype=int)


def run_adversarial_validation(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    x_tr = train_df[config.FEATURES].values
    x_va = val_df[config.FEATURES].values
    x_adv = np.vstack([x_tr, x_va])
    y_adv = np.array([0] * len(x_tr) + [1] * len(x_va))

    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=config.RANDOM_STATE)
    adv_preds = cross_val_predict(rf, x_adv, y_adv, cv=5, method='predict_proba')[:, 1]
    auc = roc_auc_score(y_adv, adv_preds)
    return float(auc)


def split():
    df = utils.read_csv("winequality-red-with-label")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[config.FEATURES].values)
    y_labels = (df[config.LABEL] >= config.QUALITY_THRESHOLD).astype(int).values

    train_idx, val_idx, test_idx = stratified_kennard_stone(
        X_scaled, y_labels,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VALIDATION_RATIO
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Kennard-Stone Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Train good ratio: {(train_df[config.LABEL] >= config.QUALITY_THRESHOLD).mean():.4f}")
    print(f"  Val good ratio:   {(val_df[config.LABEL] >= config.QUALITY_THRESHOLD).mean():.4f}")
    print(f"  Test good ratio:  {(test_df[config.LABEL] >= config.QUALITY_THRESHOLD).mean():.4f}")

    auc = run_adversarial_validation(train_df, val_df)
    print(f"  [Adversarial Validation] Train vs Val ROC-AUC: {auc:.4f} (Obiettivo ~0.50)")

    utils.save_csv(train_df, "winequality-red-train")
    utils.save_csv(val_df, "winequality-red-validation")
    utils.save_csv(test_df, "winequality-red-test")
