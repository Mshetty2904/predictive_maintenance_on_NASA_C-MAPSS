"""
Dataset Validation
"""

import pandas as pd


class DatasetValidator:

    @staticmethod
    def validate(df: pd.DataFrame):

        report = {

            "Rows": len(df),

            "Columns": len(df.columns),

            "Missing Values": df.isnull().sum().sum(),

            "Duplicate Rows": df.duplicated().sum(),

            "Engines": df.engine_id.nunique(),

            "Maximum Cycle": df.groupby(
                "engine_id"
            )["cycle"].max().max()

        }

        return report