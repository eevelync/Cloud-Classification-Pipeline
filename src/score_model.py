import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.base import BaseEstimator

# Create a logger
logger = logging.getLogger(__name__)

def predict_proba(model: BaseEstimator, x_test: pd.DataFrame, initial_features: List[str]) -> pd.Series:
    """
    Predict class probabilities with the given model.

    Args:
        model (BaseEstimator): The trained model.
        x_test (pd.DataFrame): The test features.
        initial_features (List[str]): The initial set of features to consider.

    Returns:
        pd.Series: The predicted class probabilities.
    """
    logger.debug("Predicting class probabilities.")
    y_pred_proba = model.predict_proba(x_test[initial_features])[:, 1]
    return y_pred_proba

def predict(model: BaseEstimator, x_test: pd.DataFrame, initial_features: List[str]) -> pd.Series:
    """
    Predict classes with the given model.

    Args:
        model (BaseEstimator): The trained model.
        x_test (pd.DataFrame): The test features.
        initial_features (List[str]): The initial set of features to consider.

    Returns:
        pd.Series: The predicted classes.
    """
    logger.debug("Predicting classes.")
    y_pred = model.predict(x_test[initial_features])
    return y_pred

def score_model(test: pd.DataFrame, model: BaseEstimator, config: Dict) -> pd.DataFrame:
    """
    Score the test set with the given model.

    Args:
        test (pd.DataFrame): The test set.
        model (BaseEstimator): The trained model.
        config (Dict): The configuration dictionary.

    Returns:
        pd.DataFrame: The true and predicted classes and class probabilities.
    """
    logger.info("Scoring the model.")
    x_test = test.drop(columns="class")
    y_true = test["class"]
    y_pred_proba = predict_proba(model, x_test, config["initial_features"])
    y_pred = predict(model, x_test, config["initial_features"])

    scores = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_pred_proba, "y_pred": y_pred})
    return scores

def save_scores(scores: pd.DataFrame, scores_path: Path) -> None:
    """
    Save the scores to a CSV file.

    Args:
        scores (pd.DataFrame): The scores DataFrame.
        scores_path (Path): The path to save the CSV file.

    Returns:
        None
    """
    logger.info("Saving scores to %s", scores_path)
    try:
        scores.to_csv(scores_path, index=False)
        logger.info("Scores saved to %s", scores_path)
    except Exception as e:
        logger.error("An error occurred while saving scores to %s: %s", scores_path, e)
        raise
