from google.cloud import storage
import os
from dotenv import load_dotenv


def _upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded as {destination_blob_name}."
    )

def _download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )






def download_from_bucket(filename,saveto):
    """Downloads a blob from the bucket."""

    load_dotenv()
    bucket_name = os.getenv("GOOGLE_BUCKET_NAME")

    _download_blob(bucket_name, filename, saveto)


def upload_to_bucket(filename):
    """Uploads a file to the bucket."""

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File does not exist: {filename}")

    load_dotenv()
    bucket_name = os.getenv("GOOGLE_BUCKET_NAME")
    full_filename = filename
    file_name = os.path.basename(filename)

    _upload_blob(bucket_name,full_filename,file_name)