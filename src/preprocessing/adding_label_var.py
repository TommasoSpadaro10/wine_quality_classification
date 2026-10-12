import config
import utils


def add_label_var() -> None:
    df = utils.read_csv('winequality-red-filtered')

    utils.ensure_col_exists(df, 'quality')

    df[config.LABEL] = df['quality'] - 1

    df['acidity_ratio'] = df['volatile acidity'] / df['fixed acidity']
    df['sulfur_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    df['alcohol_density'] = df['alcohol'] * df['density']
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']

    utils.save_csv(df, 'winequality-red-with-label')
