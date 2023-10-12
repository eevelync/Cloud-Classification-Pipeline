import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def create_dataset(file_path: Path, config: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    """
    Create a dataset from the provided file path and configuration.

    Args:
        file_path: The file from which to create the dataset.
        config: The configuration dict defining columns and class ranges.

    Returns:
        A pandas DataFrame representing the dataset.
    """
    columns = config['columns']
    class_1_range = config['class_1']
    class_2_range = config['class_2']

    try:
        with file_path.open('r') as f:
            data = [[s for s in line.split(' ') if s != ''] for line in f.readlines()]
    except FileNotFoundError:
        logger.error('File not found at the provided path: %s', file_path)
        raise
    except Exception as e:
        logger.error('An error occurred while opening the file: %s', e)
        raise

    # Get first cloud class
    first_cloud = data[class_1_range[0]:class_1_range[1]]
    first_cloud = [[float(s.replace('/n', '')) for s in cloud] for cloud in first_cloud]
    first_cloud = pd.DataFrame(first_cloud, columns=columns)
    first_cloud['class'] = np.zeros(len(first_cloud))

    # Get second cloud class
    second_cloud = data[class_2_range[0]:class_2_range[1]]
    second_cloud = [[float(s.replace('/n', '')) for s in cloud] for cloud in second_cloud]
    second_cloud = pd.DataFrame(second_cloud, columns=columns)
    second_cloud['class'] = np.ones(len(second_cloud))

    # Concatenate dataframes for training
    data = pd.concat([first_cloud, second_cloud])

    return data

def save_dataset(data: pd.DataFrame, save_path: Path) -> None:
    """
    Save the provided DataFrame to the specified path.

    Args:
        data: The pandas DataFrame to save.
        save_path: The path to which the DataFrame should be saved.
    """
    try:
        data.to_csv(save_path, index=False)
        logger.info('Data successfully saved to %s', save_path)
    except FileNotFoundError:
        logger.error('File not found at the provided path: %s', save_path)
        raise
    except Exception as e:
        logger.error('An error occurred while trying to save the file: %s', e)
        raise
