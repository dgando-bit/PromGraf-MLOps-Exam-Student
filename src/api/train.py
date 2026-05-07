"""
train.py — Standalone training script.

Usage:
    python train.py

Or via Makefile:
    make train

This script fetches the Bike Sharing dataset, trains a RandomForestRegressor
on January 2011 (the reference period), and persists:
  - /models/bike_model.pkl          ← the fitted model
  - /models/reference_data.parquet  ← features + target + in-sample predictions
"""

import logging
import sys

# Re-use every helper defined in main.py so there is a single source of truth.
from main import train_and_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Starting training pipeline ===")
    try:
        train_and_save()
        logger.info("=== Training pipeline completed successfully ===")
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()