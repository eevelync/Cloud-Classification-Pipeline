import logging
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

def get_data(url: str, attempts: int = 4, wait: int = 3, wait_multiple: int = 2) -> bytes:
    """
    Acquires data from the provided URL.

    Args:
        url: The URL from which to fetch the data.
        attempts: The number of attempts to make before giving up.
        wait: The initial wait time between attempts in seconds.
        wait_multiple: The factor by which the wait time is multiplied after each attempt.

    Returns:
        The fetched data as bytes.

    Raises:
        SystemExit: If the data could not be fetched after the specified number of attempts.
    """
    for _ in range(attempts):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.content
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            logger.warning("Error encountered while fetching data from %s: %s", url, e)
            time.sleep(wait)
            wait *= wait_multiple
    logger.error("Failed to fetch data from %s after %d attempts", url, attempts)
    sys.exit(1)

def write_data(data: bytes, save_path: Path) -> None:
    """
    Writes the provided data to the specified path.

    Args:
        data: The data to be written.
        save_path: The path to which the data should be written.

    Raises:
        SystemExit: If the file could not be opened for writing.
    """
    try:
        with save_path.open("wb") as f:
            f.write(data)
            logger.info("Data written to %s", save_path)
    except FileNotFoundError:
        logger.error("Please provide a valid file location to save dataset to.")
        sys.exit(1)
    except IOError as e:
        logger.error("Error occurred while trying to write dataset to file: %s", e)
        sys.exit(1)

def acquire_data(url: str, save_path: Path) -> None:
    """
    Acquires data from specified URL

    Args:
        url: URL for where data to be acquired is stored
        save_path: Local path to write data to
    
    Raises:
        SystemExit: If the data could not be fetched or written to the file.
    """
    url_contents = get_data(url)
    try:
        write_data(url_contents, save_path)
        logger.info("Data written to %s", save_path)
    except FileNotFoundError:
        logger.error("Please provide a valid file location to save dataset to.")
        sys.exit(1)
    except IOError as e:
        logger.error("Error occurred while trying to write dataset to file: %s", e)
        sys.exit(1)
