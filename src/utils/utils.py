import os
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch

import config


def ensure_dir_exists(dir_path: str):
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory '{dir_path}' does not exist.")


def read_csv(file_name: str) -> pd.DataFrame:
    ensure_dir_exists(config.DATA_DIR)
    return pd.read_csv(
        os.path.join(config.DATA_DIR, f'{file_name}.csv'), sep=';'
    )


def save_csv(df: pd.DataFrame, file_name: str):
    ensure_dir_exists(config.DATA_DIR)
    df.to_csv(
        os.path.join(config.DATA_DIR, f'{file_name}.csv'), index=False, sep=';'
    )


def ensure_col_exists(df: pd.DataFrame, col_name: str):
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in df.")


def print_descriptive_statistics(df: pd.DataFrame):
    print(df.describe(), '\n')


def make_plot(title: str, file_name: str, plot: Callable, xlabel: str = '', ylabel: str = ''):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plot()

    save_plot(file_name)


def save_plot(file_name: str):
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.PLOTS_DIR, f'{file_name}.png'))
    plt.close()


def save_model(model: torch.nn.Module, model_name: str):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(
            config.MODELS_DIR,
            f'{model_name}_{config.MODEL_VERSION}.pth'
        ))


def load_model(model: torch.nn.Module, model_name: str):
    ensure_dir_exists(config.MODELS_DIR)
    model.load_state_dict(torch.load(
        os.path.join(
            config.MODELS_DIR,
            f'{model_name}_{config.MODEL_VERSION}.pth'
        ),
        weights_only=True
    ))


def split_df_for_inference(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    input = torch.from_numpy(
        df[config.FEATURES].to_numpy().copy()
    ).float()
    labels = torch.from_numpy(
        df[config.LABEL].to_numpy().copy()
    ).long()

    return input, labels


def calculate_accuracy(num_correct: int, dataset_len: int) -> float:
    return round(num_correct / dataset_len, 4)
