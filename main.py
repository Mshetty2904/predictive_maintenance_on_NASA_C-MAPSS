"""
Main File
"""

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.validator import DatasetValidator

from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():

    logger = setup_logger()

    config = load_config()

    loader = DataLoader(
        config["dataset_path"]
    )

    datasets = [
        "FD001",
        "FD002",
        "FD003",
        "FD004"
    ]

    logger.info("=" * 60)

    logger.info(
        "NASA C-MAPSS DATA SUMMARY"
    )

    logger.info("=" * 60)

    for dataset in datasets:

        train = loader.load_train(dataset)

        report = DatasetValidator.validate(train)

        logger.info(f"\n{dataset}")

        for key, value in report.items():

            logger.info(
                f"{key:<20}: {value}"
            )


if __name__ == "__main__":
    main()