"""
Dataset Statistics Module

Author: Mayur Shetty
"""

import pandas as pd


class DatasetStatistics:

    @staticmethod
    def generate(df: pd.DataFrame):

        summary = {
            "Rows": len(df),
            "Columns": len(df.columns),
            "Missing Values": df.isnull().sum().sum(),
            "Duplicate Rows": df.duplicated().sum(),
            "Memory (MB)": round(
                df.memory_usage(deep=True).sum() / (1024 ** 2),
                2,
            ),
            "Engines": df["engine_id"].nunique(),
            "Minimum Cycle": df["cycle"].min(),
            "Maximum Cycle": df["cycle"].max(),
        }

        return pd.DataFrame([summary])