from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

import config
import utils


def plot_class_distribution(df: pd.DataFrame):
    plt.hist(df[config.LABEL], bins=config.NUM_CLASSES, edgecolor='black')


def plot_split_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    datasets = {'Train': train_df, 'Validation': val_df, 'Test': test_df}
    labels = list(datasets.keys())
    count_0 = [int((df[config.LABEL] < config.QUALITY_THRESHOLD).sum()) for df in datasets.values()]
    count_1 = [int((df[config.LABEL] >= config.QUALITY_THRESHOLD).sum()) for df in datasets.values()]

    x = range(len(labels))
    width = 0.35

    bars_0 = plt.bar([i - width / 2 for i in x], count_0, width,
                     label='0', color='red', alpha=0.6, edgecolor='black')
    bars_1 = plt.bar([i + width / 2 for i in x], count_1, width,
                     label='1', color='blue', alpha=0.6, edgecolor='black')

    for bar in bars_0:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(int(bar.get_height())),
                 ha='center', va='bottom', fontsize=9)
    for bar in bars_1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(int(bar.get_height())),
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(list(x), labels)
    plt.legend()


def plot_feature_correlation(df: pd.DataFrame):
    num_rows, num_cols = len(config.FEATURES), len(config.FEATURES)

    _, axes = plt.subplots(
        num_rows, num_cols-1, figsize=(num_rows*2, (num_cols-1)*2), constrained_layout=True
    )

    def get_axes(r: int, c: int) -> Any:
        if c > r:
            return axes[r, c-1]
        else:
            return axes[r, c]

    colors = list(map(
        lambda v: 'blue' if v >= config.QUALITY_THRESHOLD else 'red',
        df[config.LABEL]
    ))

    for r in range(num_rows):
        for c in range(num_cols):
            if r != c:
                ax = get_axes(r, c)

                ax.scatter(
                    df[config.FEATURES[r]],
                    df[config.FEATURES[c]],
                    c=colors
                )

                ax.set_xlabel(config.FEATURES[r])
                ax.set_ylabel(config.FEATURES[c])


def plot_feature_vs_target(df: pd.DataFrame):
    num_features = len(config.FEATURES)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), constrained_layout=True
    )

    group_0 = df[df[config.LABEL] < config.QUALITY_THRESHOLD]
    group_1 = df[df[config.LABEL] >= config.QUALITY_THRESHOLD]

    for i, feature in enumerate(config.FEATURES):
        r, c = divmod(i, num_cols)
        ax = axes[r, c]

        bp = ax.boxplot(
            [group_0[feature], group_1[feature]],
            labels=['0', '1'],
            patch_artist=True
        )

        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('blue')

        for box in bp['boxes']:
            box.set_alpha(0.6)

        ax.set_xlabel(config.LABEL)
        ax.set_ylabel(feature)

    # Nasconde gli assi vuoti se il numero di feature non riempie la griglia
    for i in range(num_features, num_rows * num_cols):
        r, c = divmod(i, num_cols)
        axes[r, c].set_visible(False)


def describe_processed_data():
    train_df = utils.read_csv('winequality-red-train')
    val_df = utils.read_csv('winequality-red-validation')
    test_df = utils.read_csv('winequality-red-test')

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    utils.ensure_col_exists(df, config.LABEL)

    print("Descriptive statistics for processed data:", '\n')
    utils.print_descriptive_statistics(df)

    utils.make_plot(
        title='Processed Wine Label Distribution',
        xlabel='Label',
        ylabel='Frequency',
        file_name='wine_label_distribution',
        plot=lambda: plot_class_distribution(df)
    )

    utils.make_plot(
        title='Label Distribution per Split',
        xlabel='Dataset',
        ylabel='Count',
        file_name='wine_label_split_distribution',
        plot=lambda: plot_split_class_distribution(train_df, val_df, test_df)
    )

    utils.make_plot(
        title='Processed Wine Label Correlation',
        file_name='wine_label_correlation',
        plot=lambda: plot_feature_correlation(df)
    )

    utils.make_plot(
        title='Feature vs Target',
        file_name='feature_vs_target',
        plot=lambda: plot_feature_vs_target(df)
    )
