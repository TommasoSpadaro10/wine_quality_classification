import config
import utils


def standardize():
    train_df = utils.read_csv('winequality-red-train')
    validation_df = utils.read_csv('winequality-red-validation')
    test_df = utils.read_csv('winequality-red-test')

    for col in config.FEATURES:
        utils.ensure_col_exists(train_df, col)

    train_mean = train_df[config.FEATURES].mean()
    train_std = train_df[config.FEATURES].std()

    train_df[config.FEATURES] = (train_df[config.FEATURES] - train_mean) / train_std
    validation_df[config.FEATURES] = (validation_df[config.FEATURES] - train_mean) / train_std
    test_df[config.FEATURES] = (test_df[config.FEATURES] - train_mean) / train_std

    utils.save_csv(train_df, 'winequality-red-train')
    utils.save_csv(validation_df, 'winequality-red-validation')
    utils.save_csv(test_df, 'winequality-red-test')
