import logging
from pathlib import Path
from typing import List, Dict

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Create a logger
logger = logging.getLogger(__name__)

def upload_artifacts(artifacts: Path, config: Dict) -> List[str]:
    """
    Upload artifacts to AWS S3.

    Args:
        artifacts (Path): The local path of the artifacts.
        config (Dict): The configuration dictionary.

    Returns:
        List[str]: The S3 URIs of the uploaded artifacts.
    """
    logger.info("Uploading artifacts to S3.")
    s3 = boto3.client("s3")
    bucket_name = config["bucket_name"]
    prefix = config["prefix"]
    uploaded_uris = []

    for file in artifacts.glob("**/*"):
        if file.is_file():
            key = f"{prefix}/{file.relative_to(artifacts)}"
            try:
                s3.upload_file(str(file), bucket_name, key)
                logger.debug("Uploaded file %s to S3 bucket %s with key %s", file, bucket_name, key)
                uploaded_uris.append(f"s3://{bucket_name}/{key}")
            except (BotoCoreError, ClientError) as e:
                logger.error("An error occurred while uploading file %s to S3: %s", file, e)
                raise

    logger.info("Finished uploading artifacts to S3.")
    return uploaded_uris
