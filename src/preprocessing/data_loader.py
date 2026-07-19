"""
NASA C-MAPSS Dataset Loader
"""

from pathlib import Path

import pandas as pd

COLUMN_NAMES = (
    ["engine_id",
     "cycle",
     "op_setting_1",
     "op_setting_2",
     "op_setting_3"]
    +
    [f"sensor_{i}" for i in range(1, 22)]
)


class DataLoader:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def load_train(self, fd):

        path = (
            self.dataset_path
            / fd
            / f"train_{fd}.txt"
        )

        return self._read(path)

    def load_test(self, fd):

        path = (
            self.dataset_path
            / fd
            / f"test_{fd}.txt"
        )

        return self._read(path)

    def load_rul(self, fd):

        path = (
            self.dataset_path
            / fd
            / f"RUL_{fd}.txt"
        )

        return pd.read_csv(
            path,
            header=None,
            names=["RUL"]
        )

    def _read(self, path):

        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None
        )

        df.dropna(
            axis=1,
            how="all",
            inplace=True
        )

        df.columns = COLUMN_NAMES

        return df