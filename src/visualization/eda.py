"""
EDA Visualizations
"""

from pathlib import Path

import matplotlib.pyplot as plt


class EDAPlots:

    @staticmethod
    def engine_cycle_distribution(df, dataset_name):

        output = Path("figures") / dataset_name

        output.mkdir(parents=True, exist_ok=True)

        engine_cycles = (
            df.groupby("engine_id")["cycle"]
            .max()
        )

        plt.figure(figsize=(10,6))

        plt.hist(engine_cycles, bins=20)

        plt.title(
            f"{dataset_name} Engine Life Distribution"
        )

        plt.xlabel("Maximum Cycle")

        plt.ylabel("Number of Engines")

        plt.tight_layout()

        plt.savefig(
            output / "engine_life_distribution.png",
            dpi=300,
        )

        plt.close()

    @staticmethod
    def sensor_distribution(df, dataset_name):

        output = Path("figures") / dataset_name

        output.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10,6))

        plt.hist(df["sensor_2"], bins=40)

        plt.title(
            f"{dataset_name} Sensor 2 Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            output / "sensor2_distribution.png",
            dpi=300,
        )

        plt.close()