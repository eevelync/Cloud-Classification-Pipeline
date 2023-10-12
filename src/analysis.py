import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Create a logger
LOGGER = logging.getLogger(__name__)

def plot_histograms(features: pd.DataFrame, output_path: Path) -> None:
    """
    Plots histograms for each feature in a DataFrame, excluding the "class" feature.
    
    :param features: DataFrame containing the features.
    :param output_path: Path to save the histogram images.
    """
    LOGGER.info("Starting to plot histograms.")
    target = features["class"]
    for feat in features.columns:
        if feat != "class":
            LOGGER.debug("Plotting histogram for %s", feat)
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.hist([
                    features[target == 0][feat].values, features[target == 1][feat].values
                ])
                ax.set_xlabel(" ".join(feat.split("_")).capitalize())
                ax.set_ylabel("Number of observations")
                fig_filename = output_path / f"{feat}_histogram.png"
                fig.savefig(fig_filename)
                LOGGER.info("Histogram saved to %s", fig_filename)
            except (OSError, FileNotFoundError) as e:
                LOGGER.error("Failed to plot or save histogram for %s. Error: %s", feat, e)
            finally:
                plt.close(fig)

def save_figures(data: pd.DataFrame, figures_path: Path) -> None:
    """
    Save histograms for each feature in a DataFrame to a specified path.
    
    :param data: DataFrame containing the features.
    :param figures_path: Path to save the histogram images.
    """
    LOGGER.info("Saving figures to %s", figures_path)
    try:
        plot_histograms(data, figures_path)
        LOGGER.info("Figures saved successfully.")
    except (OSError, FileNotFoundError) as e:
        LOGGER.error("Failed to save figures. Error: %s", e)
