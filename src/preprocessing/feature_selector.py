"""
Feature Selection Module

Automatically removes uninformative features.
"""

import pandas as pd


class FeatureSelector:
    """
    Handles feature selection.
    """

    @staticmethod
    def select(
        df: pd.DataFrame,
        remove_constant: bool = True,
        variance_threshold: float = 0.001,
        manual_remove: list | None = None,
    ) -> pd.DataFrame:
        data = df.copy()
        #Don't touch the protected columns
        protected = [
            "engine_id",
            "cycle",
            "RUL"
        ]

        #Find Sensor columns
        feature_columns = [
            col
            for col in data.columns
            if col not in protected
        ]

        #Remove constant features
        if remove_constant:
            constant = [
                col
                for col in feature_columns
                if data[col].nunique() == 1
            ]

            data.drop(columns=constant, inplace=True)

            feature_columns = [
                c
                for c in feature_columns
                if c not in constant
            ]
        
        #Remove low variance features
        low_variance = [
            col
            for col in feature_columns
            if data[col].var() < variance_threshold
        ]

        data.drop(columns=low_variance, inplace=True)

        #Manually remove features if specified
        if manual_remove:
            removable = [
                c
                for c in manual_remove
                if c in data.columns
            ]

            data.drop(columns=removable, inplace=True)
        return data
    
