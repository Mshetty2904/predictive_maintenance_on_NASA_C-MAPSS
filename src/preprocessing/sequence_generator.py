"""
Sequence Generation Module

Creates sliding-window sequences for time-series models.
"""

import numpy as np
import pandas as pd


class SequenceGenerator:
    """
    Generate sliding-window sequences for predictive maintenance.
    """

    EXCLUDED_COLUMNS = (
        "engine_id",
        "cycle",
        "RUL",
    )

    def __init__(
        self,
        window_size: int = 30,
        stride: int = 1,
    ):
        self.window_size = window_size
        self.stride = stride

    @classmethod
    def get_feature_columns(
        cls,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Return feature columns used for sequence generation.
        """
        return [
            column
            for column in df.columns
            if column not in cls.EXCLUDED_COLUMNS
        ]
    def generate_engine_sequences(
        self,
        engine_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate sliding-window sequences for a single engine.

        Parameters
        ----------
        engine_df : pd.DataFrame
            Data for one engine only.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            X sequences and corresponding RUL labels.
        """

        feature_columns = self.get_feature_columns(engine_df)

        X = []
        y = []

        feature_values = engine_df[feature_columns].to_numpy()

        rul_values = engine_df["RUL"].to_numpy()

        total_cycles = len(engine_df)

        for start in range(
            0,
            total_cycles - self.window_size + 1,
            self.stride,
        ):

            end = start + self.window_size

            X.append(
                feature_values[start:end]
            )

            y.append(
                rul_values[end - 1]
            )

        return (
            np.array(X),
            np.array(y),
        )
    
    def generate(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate sliding-window sequences for all engines.

        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset containing multiple engines.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Feature sequences (X) and target RUL values (y).
        """

        X_all = []
        y_all = []

        for engine_id in sorted(df["engine_id"].unique()):

            engine_df = (
                df[df["engine_id"] == engine_id]
                .sort_values("cycle")
                .reset_index(drop=True)
            )

            X_engine, y_engine = self.generate_engine_sequences(
                engine_df
            )

            if len(X_engine) > 0:
                X_all.append(X_engine)
                y_all.append(y_engine)

        if not X_all:
            raise ValueError(
                "No sequences could be generated. "
                "Check the window size."
            )

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

        return X, y
    @staticmethod
    def get_shapes(
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Return sequence shapes.
        """

        return {
            "X Shape": X.shape,
            "y Shape": y.shape,
            "Samples": len(X),
            "Window Size": X.shape[1],
            "Features": X.shape[2],
    }