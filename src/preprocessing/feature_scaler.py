"""
Feature Scaling Module

Scales numerical features for machine learning models.

Supported Scalers:
- StandardScaler
- MinMaxScaler
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)


class FeatureScaler:
    """
    Feature scaling utility for machine learning models.

    This class provides a unified interface for fitting,
    transforming, saving, and loading feature scalers.
    """

    EXCLUDED_COLUMNS = (
        "engine_id",
        "cycle",
        "RUL",
    )

    def __init__(
        self,
        method: str = "standard",
    ) -> None:

        self.method = method.lower()
        self.feature_columns: list[str] = []

        if self.method == "standard":
            self.scaler = StandardScaler()

        elif self.method == "minmax":
            self.scaler = MinMaxScaler()

        else:
            raise ValueError(
                f"Unsupported scaling method: {method}"
            )

    @classmethod
    def get_feature_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:

        return [
            column
            for column in df.columns
            if column not in cls.EXCLUDED_COLUMNS
        ]

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "FeatureScaler":

        self.feature_columns = self.get_feature_columns(df)

        if not self.feature_columns:
            raise ValueError(
                "No feature columns available for scaling."
            )

        self.scaler.fit(
            df[self.feature_columns]
        )

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not self.feature_columns:
            raise RuntimeError(
                "Scaler must be fitted before calling transform()."
            )

        data = df.copy()

        data[self.feature_columns] = self.scaler.transform(
            data[self.feature_columns]
        )

        return data

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.scaler,
            path,
        )

    def load(
        self,
        path: str | Path,
    ) -> "FeatureScaler":
        path = Path(path)

        self.scaler = joblib.load(path)

        return self