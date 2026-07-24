"""
Preprocessing Pipeline

Coordinates all preprocessing operations.

Author: Mayur Shetty
"""

from pathlib import Path

from src.preprocessing.feature_scaler import FeatureScaler
from src.preprocessing.feature_selector import FeatureSelector
from src.preprocessing.rul_generator import RULGenerator


class PreprocessingPipeline:
    """
    Coordinates all preprocessing steps.
    """

    def __init__(self, config, logger):
        """
        Initialize preprocessing pipeline.
        """

        self.config = config
        self.logger = logger

    def generate_rul(self, train):
        """
        Generate Remaining Useful Life (RUL) labels.
        """

        train = RULGenerator.generate(
            train,
            cap_enabled=self.config["rul"]["cap_enabled"],
            cap_value=self.config["rul"]["cap_value"],
        )

        self.logger.info(
            "RUL labels generated successfully."
        )

        self.logger.info(
            "RUL Range : %d -> %d",
            train["RUL"].min(),
            train["RUL"].max(),
        )

        return train

    def feature_selection(self, train):
        """
        Remove constant and low-variance features.
        """

        train = FeatureSelector.select(
            train,
            remove_constant=self.config[
                "feature_selection"
            ]["remove_constant"],
            variance_threshold=self.config[
                "feature_selection"
            ]["variance_threshold"],
            manual_remove=self.config[
                "feature_selection"
            ]["manual_remove"],
        )

        self.logger.info(
            "Feature Selection completed."
        )

        self.logger.info(
            "Remaining Columns : %d",
            len(train.columns),
        )

        return train

    def scale_features(
        self,
        train,
        dataset,
    ):
        """
        Scale feature columns using the configured scaler.
        """

        scaling_config = self.config["scaling"]

        if not scaling_config["enabled"]:

            self.logger.info(
                "Feature scaling skipped."
            )

            return train

        scaler = FeatureScaler(
            method=scaling_config["method"]
        )

        train = scaler.fit_transform(train)

        self.logger.info(
            "Feature Scaling completed."
        )

        self.logger.info(
            "Scaling Method : %s",
            scaling_config["method"],
        )

        if scaling_config["save_scaler"]:

            scaler_dir = Path(
                scaling_config["scaler_path"]
            )

            scaler_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            scaler_path = (
                scaler_dir /
                f"{dataset}_scaler.pkl"
            )

            scaler.save(
                scaler_path
            )

            self.logger.info(
                "Scaler saved : %s",
                scaler_path,
            )

        return train

    def run(
        self,
        train,
        dataset,
    ):
        """
        Execute the preprocessing pipeline.
        """

        train = self.generate_rul(train)

        train = self.feature_selection(train)

        train = self.scale_features(
            train,
            dataset,
        )

        return train