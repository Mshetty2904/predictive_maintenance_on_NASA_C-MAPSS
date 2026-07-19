"""
Main File

Predictive Maintenance using NASA C-MAPSS
Author: Mayur Shetty
"""

from pathlib import Path
import time
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.validator import DatasetValidator
from src.preprocessing.statistics import DatasetStatistics
from src.preprocessing.sensor_analysis import SensorAnalysis

from src.visualization.eda import EDAPlots

from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    """
    Main execution pipeline.
    """

    logger = setup_logger()

    logger.info("=" * 70)
    logger.info("PREDICTIVE MAINTENANCE PIPELINE")
    logger.info("=" * 70)
    start_time = time.time()

    try:
        # ---------------------------------------------------------
        # Load Configuration
        # ---------------------------------------------------------
        config = load_config()

        loader = DataLoader(config["dataset_path"])

        datasets = config["datasets"]

        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        logger.info("Datasets to process: %s", ", ".join(datasets))

        # ---------------------------------------------------------
        # Process Each Dataset
        # ---------------------------------------------------------
        for dataset in datasets:

            logger.info("\n%s", "=" * 60)
            logger.info("Processing Dataset: %s", dataset)
            logger.info("%s", "=" * 60)
            try:
                # -----------------------------------------
                # Load Dataset
                # -----------------------------------------
                train = loader.load_train(dataset)

                # -----------------------------------------
                # Validation
                # -----------------------------------------
                report = DatasetValidator.validate(train)

                logger.info("Dataset Validation")

                for key, value in report.items():
                    logger.info(f"{key:<22}: {value}")

                # -----------------------------------------
                # Dataset Statistics
                # -----------------------------------------
                stats = DatasetStatistics.generate(train)

                stats.to_csv(
                    output_dir / f"{dataset}_summary.csv",
                    index=False,
                )

                logger.info("Dataset summary saved.")

                # -----------------------------------------
                # Sensor Statistics
                # -----------------------------------------
                sensor_stats = SensorAnalysis.sensor_statistics(train)

                sensor_stats.to_csv(
                    output_dir / f"{dataset}_sensor_statistics.csv"
                )

                logger.info("Sensor statistics saved.")

                # -----------------------------------------
                # Constant Sensors
                # -----------------------------------------
                constant = SensorAnalysis.constant_sensors(train)

                logger.info(
                    "Constant Sensors (%d): %s",
                    len(constant),
                    constant if constant else "None",
                )

                # -----------------------------------------
                # Low Variance Sensors
                # -----------------------------------------
                low = SensorAnalysis.low_variance(train)

                logger.info(
                    "Low Variance Sensors (%d):",
                    len(low)
                )

                if low.empty:
                    logger.info("None")

                else:
                    for sensor, variance in low.items():
                        logger.info(
                            "%-12s : %.8f",
                            sensor,
                            variance
                        )

                # -----------------------------------------
                # Visualizations
                # -----------------------------------------
                EDAPlots.engine_cycle_distribution(
                    train,
                    dataset,
                )

                EDAPlots.sensor_distribution(
                    train,
                    dataset,
                )

                logger.info("EDA figures generated successfully.")

                logger.info(
                    "Dataset %s processed successfully.",
                    dataset,
                )

            except Exception:
                logger.exception(
                    "Failed while processing dataset %s",
                    dataset,
                )

        logger.info("\n%s", "=" * 70)
        logger.info("Pipeline completed successfully.")
        logger.info("%s", "=" * 70)

    except Exception:
        logger.exception(
            "Fatal error while initializing the pipeline."
        )
    finally:
        elapsed = time.time() - start_time

        logger.info(
            "Total Execution Time : %.2f seconds",
            elapsed
        )

if __name__ == "__main__":
    main()