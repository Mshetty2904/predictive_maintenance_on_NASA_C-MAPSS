"""
RUL Generator

Generates Remaining Useful Life (RUL) for NASA C-MAPSS datasets.
"""

import pandas as pd
from xgboost import data


class RULGenerator:
    """
    Generates Remaining Useful Life (RUL) labels.
    """

    @staticmethod
    def generate(
        df: pd.DataFrame,
        cap_enabled: bool = False,
        cap_value: int = 125,
    ) -> pd.DataFrame:

        # Create a copy to avoid modifying the original DataFrame
        data = df.copy()

        # Maximum cycle for each engine
        max_cycle = (
            data.groupby("engine_id")["cycle"]
            .max()
            .rename("max_cycle")
        )

        # Merge with original dataset
        data = data.merge(
            max_cycle,
            on="engine_id",
            how="left"
        )

        # Compute Remaining Useful Life
        data["RUL"] = data["max_cycle"] - data["cycle"]
        if cap_enabled:    
            data["RUL"] = data["RUL"].clip(upper=cap_value)

        # Remove helper column
        data.drop(columns=["max_cycle"], inplace=True)

        return data