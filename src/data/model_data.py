"""
Model Data Preparation

Prepare datasets for machine learning models.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing.sequence_generator import (
    SequenceGenerator,
)


class ModelData:
    """
    Prepare datasets for machine learning models.
    """

    EXCLUDED_COLUMNS = (
        "engine_id",
        "cycle",
        "RUL",
    )

    def __init__(
        self,
        config,
    ):
        """
        Initialize ModelData.

        Parameters
        ----------
        config : dict
            Project configuration.
        """
        self.config = config

    def prepare_xgboost(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare tabular data for XGBoost.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Feature matrix (X) and target (y).
        """

        feature_columns = [
            column
            for column in df.columns
            if column not in self.EXCLUDED_COLUMNS
        ]

        X = df[feature_columns]

        y = df["RUL"]

        return X, y

    def prepare_sequences(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM/CNN-LSTM.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Feature sequences (X) and targets (y).
        """

        sequence_config = self.config["sequence"]

        generator = SequenceGenerator(
            window_size=sequence_config["window_size"],
            stride=sequence_config["stride"],
        )

        return generator.generate(df)

    def split_data(
        self,
        X,
        y,
    ):
        """
        Split dataset into training and testing sets.

        Parameters
        ----------
        X
            Features.
        y
            Target values.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """

        split_config = self.config["train_test_split"]

        return train_test_split(
            X,
            y,
            test_size=split_config["test_size"],
            random_state=split_config["random_state"],
            shuffle=split_config["shuffle"],
        )

    @staticmethod
    def summary(
        X,
        y,
    ) -> dict:
        """
        Return dataset summary.

        Parameters
        ----------
        X
            Features.

        y
            Target values.

        Returns
        -------
        dict
            Dataset information.
        """

        return {
            "Samples": len(X),
            "Features": X.shape[-1] if hasattr(X, "shape") else None,
            "Target Shape": y.shape,
        }