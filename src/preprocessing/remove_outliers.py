import pandas as pd
from typing import cast
import config
import utils


def remove_outliers() -> None:
    df = utils.read_csv('winequality-red')

    rows_before = len(df)

    for feature, (low, high) in config.PLAUSIBLE_RANGES.items():
        utils.ensure_col_exists(df, feature)
        df = cast(pd.DataFrame, df[(df[feature] >= low) & (df[feature] <= high)])

    rows_after = len(df)
    removed = rows_before - rows_after

    print(f"Righe prima del filtraggio: {rows_before}")
    print(f"Righe dopo il filtraggio:   {rows_after}")
    print(f"Righe rimosse:              {removed} ({removed / rows_before * 100:.1f}%)")

    utils.save_csv(df, 'winequality-red-filtered')

