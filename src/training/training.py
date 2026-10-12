from typing import cast

import torch

import config
import utils
from model import WineQualityClassifier
from .loss import CostSensitiveLoss, NegativeCorrelationLoss, LatentCosineOrthogonalityLoss


def get_train_loader() -> tuple[torch.utils.data.DataLoader, int]:
    df = utils.read_csv("winequality-red-train")

    input, labels = utils.split_df_for_inference(df)
    labels = labels.long()

    input = input.to(config.DEVICE)
    labels = labels.to(config.DEVICE)

    dataset_len = input.shape[0]

    dataset = torch.utils.data.TensorDataset(input, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=False
    )

    return loader, dataset_len


def get_val_data() -> tuple[torch.Tensor, torch.Tensor]:
    df = utils.read_csv("winequality-red-validation")
    return utils.split_df_for_inference(df)


def train() -> None:
    print("\n--- Starting Joint Hybrid Training: NCL + Latent Orthogonality (H1 _|_ H2) ---", flush=True)
    model1 = WineQualityClassifier().to(config.DEVICE)
    model2 = WineQualityClassifier().to(config.DEVICE)

    cost_loss_fn = CostSensitiveLoss()
    ncl_loss_fn = NegativeCorrelationLoss()
    ortho_loss_fn = LatentCosineOrthogonalityLoss()

    optimizer = torch.optim.Adam(
        list(model1.parameters()) + list(model2.parameters()),
        lr=config.LEARNING_RATE,
        betas=config.ADAM_BETAS,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=1e-6
    )

    train_loader, dataset_len = get_train_loader()
    val_input, val_labels = get_val_data()
    val_input = val_input.to(config.DEVICE)
    val_labels = val_labels.to(config.DEVICE)
    val_labels_good = val_labels >= config.QUALITY_THRESHOLD

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(config.EPOCHS):
        model1.train()
        model2.train()
        num_correct = 0
        total_loss = 0.0

        for input, labels in train_loader:
            input = cast(torch.Tensor, input)
            labels = cast(torch.Tensor, labels)

            optimizer.zero_grad()
            h1 = model1.layers[:8](input)
            p1 = model1.layers[8:](h1)
            h2 = model2.layers[:8](input)
            p2 = model2.layers[8:](h2)

            l1 = cost_loss_fn(p1, labels)
            l2 = cost_loss_fn(p2, labels)
            cov_penalty = ncl_loss_fn(p1, p2, labels)
            ortho_penalty = ortho_loss_fn(h1, h2)

            batch_loss = (
                l1 + l2
                + config.NCL_LAMBDA * cov_penalty
                + config.ORTHO_LAMBDA * ortho_penalty
            )
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

            p_ens = 0.5 * (p1 + p2)
            pred_good = p_ens.argmax(dim=1) >= config.QUALITY_THRESHOLD
            labels_good = labels >= config.QUALITY_THRESHOLD
            num_correct += (pred_good == labels_good).sum().item()

        # Validation on Ensemble
        model1.eval()
        model2.eval()
        with torch.no_grad():
            vh1 = model1.layers[:8](val_input)
            vp1 = model1.layers[8:](vh1)
            vh2 = model2.layers[:8](val_input)
            vp2 = model2.layers[8:](vh2)
            vp_ens = 0.5 * (vp1 + vp2)
            val_pred_good = vp_ens.argmax(dim=1) >= config.QUALITY_THRESHOLD
            val_correct = (val_pred_good == val_labels_good).sum().item()

        train_acc = utils.calculate_accuracy(num_correct, dataset_len)
        val_acc = utils.calculate_accuracy(val_correct, val_input.shape[0])

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            utils.save_model(model1, 'wine_quality_model_net1')
            utils.save_model(model2, 'wine_quality_model_net2')
            utils.save_model(model1, 'wine_quality_model')
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if (
            config.EARLY_STOPPING_ENABLED
            and best_val_accuracy >= config.EARLY_STOPPING_MIN_ACCURACY
            and epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE
        ):
            print(f"[Joint Hybrid Model] Early stopping at epoch {epoch + 1} (best Val Accuracy: {best_val_accuracy:.4f})", flush=True)
            break

    print(f"[Joint Hybrid Model] Training finished. Best Val Accuracy: {best_val_accuracy:.4f}", flush=True)
    utils.load_model(model1, 'wine_quality_model_net1')
    utils.load_model(model2, 'wine_quality_model_net2')
    with torch.no_grad():
        h1_val = model1.get_latent(val_input)
        h2_val = model2.get_latent(val_input)
        val_ortho = ortho_loss_fn(h1_val, h2_val).item()
    print(f"Mutual Latent Orthogonality on Validation (cos^2): {val_ortho:.4f}", flush=True)
    print("Joint Hybrid NCL + Latent Orthogonality training completed successfully.", flush=True)

