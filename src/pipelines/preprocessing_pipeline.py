"""
Preprocessing Pipeline

Coordinates all preprocessing operations.

Author: Mayur Shetty
"""

from xgboost import train

from src.preprocessing.rul_generator import RULGenerator
from src.preprocessing.feature_selector import FeatureSelector


class PreprocessingPipeline:
    """
    Coordinates preprocessing steps.
    """

    def __init__(self, config, logger):

        self.config = config
        self.logger = logger
    def generate_rul(self, train):

        train = RULGenerator.generate(
            train,
            cap_enabled=self.config["rul"]["cap_enabled"],
            cap_value=self.config["rul"]["cap_value"],
        )

        self.logger.info("RUL labels generated successfully.")

        self.logger.info(
            "RUL Range : %d -> %d",
            train["RUL"].min(),
            train["RUL"].max(),
        )

        return train
    def feature_selection(self, train):

        train = FeatureSelector.select(
            train,
            remove_constant=self.config["feature_selection"]["remove_constant"],
            variance_threshold=self.config["feature_selection"]["variance_threshold"],
            manual_remove=self.config["feature_selection"]["manual_remove"],
        )

        self.logger.info("Feature Selection completed.")

        self.logger.info(
            "Remaining Columns : %d",
            len(train.columns),
        )

        return train
    def run(self, train, dataset):

        train = self.generate_rul(train)

        train = self.feature_selection(train)

        return train
    