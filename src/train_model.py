import logging
import pickle
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
from sklearn.base import BaseEstimator

# Create a logger
logger = logging.getLogger(__name__)

def split_data(features: pd.DataFrame,
    target: pd.Series,
    test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    """
    Split features and target into training and testing sets.

    Args:
        features (pd.DataFrame): The features DataFrame.
        target (pd.Series): The target Series.
        test_size (float): The proportion of the data to include in the test split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training and testing sets.
    """
    logger.info("Splitting data into train and test sets.")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, target, test_size=test_size)
    return x_train, x_test, y_train, y_test

def train_random_forest(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int,
        max_depth: int,
        initial_features: List[str]
) -> sklearn.ensemble.RandomForestClassifier:

    """
    Train a Random Forest classifier.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        n_estimators (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the trees.
        initial_features (List[str]): The initial set of features to consider.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained Random Forest classifier.
    """
    logger.info("Training Random Forest model.")
    random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    random_forest.fit(x_train[initial_features], y_train)
    return random_forest

def train_model(data: pd.DataFrame, config: dict) -> Tuple[BaseEstimator, pd.DataFrame, pd.DataFrame]:
    """
    Train a Random Forest classifier and return the trained model, training set, and testing set.

    Args:
        data (pd.DataFrame): The input DataFrame with features and target.
        config (dict): The configuration dictionary with hyperparameters and test size.

    Returns:
        Tuple[BaseEstimator, pd.DataFrame, pd.DataFrame]: The trained model, training set, and testing set.
    """
    logger.info("Starting model training.")
    features = data.drop(columns="class")
    target = data["class"]
    x_train, x_test, y_train, y_test = split_data(features, target, config["test_size"])
    model = train_random_forest(x_train, y_train, config["n_estimators"], 
                            config["max_depth"], config["initial_features"])

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    return model, train, test

def save_data(train: pd.DataFrame, test: pd.DataFrame, artifacts: Path) -> None:
    """
    Save the training and testing sets to CSV files.

    Args:
        train (pd.DataFrame): The training set.
        test (pd.DataFrame): The testing set.
        artifacts (Path): The directory to save the CSV files in.

    Returns:
        None
    """
    logger.info("Saving train and test data.")
    try:
        train.to_csv(artifacts / "train.csv", index=False)
        test.to_csv(artifacts / "test.csv", index=False)
    except Exception as e:
        logging.error("An error occurred while saving data: %s", str(e))
        raise

def save_model(model: sklearn.base.BaseEstimator, model_path: Path) -> None:
    """
    Save a trained model to a file.

    Args:
        model: The trained model object.
        model_path: The path to save the model file.

    Raises:
        Exception: If there is an error while saving the model.
    """
    logger.info("Saving the model.")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", model_path)
    except Exception as e:
        logger.exception("Error while saving the model to %s", model_path)
        raise e
