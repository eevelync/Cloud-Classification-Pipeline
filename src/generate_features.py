"""
This module contains functions for generating features from a given dataset.
"""

import logging
from typing import Dict, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def log_transform(data: pd.DataFrame, column: str, new_column: str) -> pd.DataFrame:
    """
    Applies the natural logarithm to a column and stores the result in a new column.
    """
    logger.debug("Applying log transformation on column %s", column)
    if (data[column] <= 0).any():
        logger.error("All values in the column must be positive.")
        raise ValueError("All values in the column must be positive.")
    data[new_column] = data[column].apply(np.log)
    return data


def multiply_columns(data: pd.DataFrame, col_a: str, col_b: str, new_column: str) -> pd.DataFrame:
    """
    Multiplies two columns and stores the result in a new column.
    """
    logger.debug("Multiplying columns %s and %s", col_a, col_b)
    data[new_column] = data[col_a].multiply(data[col_b])
    return data


def calculate_norm_range(data: pd.DataFrame, min_col: str, max_col: str, mean_col: str, new_column: str) -> pd.DataFrame:
    """
    Calculates the normalized range and stores the result in a new column.
    """
    logger.debug("Calculating normalized range for columns %s, %s, %s", min_col, max_col, mean_col)
    data[new_column] = (data[max_col] - data[min_col]) / data[mean_col]
    return data


def calculate_range(data: pd.DataFrame, min_col: str, max_col: str, new_column: str) -> pd.DataFrame:
    """
    Calculates the range and stores the result in a new column.
    """
    logger.debug("Calculating range for columns %s and %s", min_col, max_col)
    data[new_column] = data[max_col] - data[min_col]
    return data

def generate_features(
    data: pd.DataFrame,
    config: Dict[str, Union[str, Dict[str, str]]]
) -> pd.DataFrame:
    """
    Generates features based on a given configuration.

    :param data: DataFrame to generate features from
    :param config: A configuration dictionary for feature generation
    :return: DataFrame with new features
    """
    logger.info("Starting to generate features.")
    features = data.copy()

    log_transform_config = config.get("log_transform", {})
    log_entropy_col = log_transform_config.get("log_entropy", "")
    if log_entropy_col:
        logger.info("Performing log_transform on column %s", log_entropy_col)
        features = log_transform(features, log_entropy_col, "log_entropy")

    multiply_config = config.get("multiply", {})
    entropy_contrast_config = multiply_config.get("entropy_x_contrast", {})
    col_a = entropy_contrast_config.get("col_a", "")
    col_b = entropy_contrast_config.get("col_b", "")
    if col_a and col_b:
        logger.info("Performing multiply_columns on columns %s and %s", col_a, col_b)
        features = multiply_columns(features, col_a, col_b, "entropy_x_contrast")

    norm_range_config = config.get("calculate_norm_range", {})
    norm_range_conf = norm_range_config.get("IR_norm_range", {})
    min_col = norm_range_conf.get("min_col", "")
    max_col = norm_range_conf.get("max_col", "")
    mean_col = norm_range_conf.get("mean_col", "")
    if min_col and max_col and mean_col:
        logger.info("Performing calculate_norm_range on columns %s, %s, and %s",
                    min_col, max_col, mean_col)
        features = calculate_norm_range(features, min_col, max_col, mean_col, "IR_norm_range")

    range_config = config.get("calculate_range", {})
    range_conf = range_config.get("IR_range", {})
    min_col = range_conf.get("min_col", "")
    max_col = range_conf.get("max_col", "")
    if min_col and max_col:
        logger.info("Performing calculate_range on columns %s and %s", min_col, max_col)
        features = calculate_range(features, min_col, max_col, "IR_range")

    logger.info("Feature generation completed.")
    return features

