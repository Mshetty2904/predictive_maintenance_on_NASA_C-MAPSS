"""
Sensor Analysis Module
"""

import pandas as pd


class SensorAnalysis:

    @staticmethod
    def sensor_statistics(df):

        sensor_cols = [c for c in df.columns if c.startswith("sensor")]

        stats = df[sensor_cols].describe().T

        stats["variance"] = df[sensor_cols].var()

        return stats

    @staticmethod
    def constant_sensors(df):

        sensor_cols = [c for c in df.columns if c.startswith("sensor")]

        constant = []

        for col in sensor_cols:

            if df[col].nunique() == 1:
                constant.append(col)

        return constant

    @staticmethod
    def low_variance(df, threshold=0.01):

        sensor_cols = [c for c in df.columns if c.startswith("sensor")]

        variances = df[sensor_cols].var()

        return variances[variances < threshold]