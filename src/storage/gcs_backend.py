import json
from typing import Optional, List
from src.storage.storage_backend import StorageBackend
from google.cloud import storage as gcs


class GCSBackend(StorageBackend):
    """
    Google Cloud Storage implementation of StorageBackend.

    This backend stores and retrieves files from a single GCS bucket.
    Object names are treated as relative paths and mirror the logical
    directory structure used by local storage.

    Prerequisites:
        Application credentials configured via one of:
            * GOOGLE_APPLICATION_CREDENTIALS environment variable
            * Explicit service account JSON passed to the constructor
    """

    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        """Initialize a Google Cloud Storage client bound to a single bucket."""

        # If credentials_path provided, use it; otherwise rely on
        # GOOGLE_APPLICATION_CREDENTIALS env var (set during setup)
        if credentials_path:
            self.client = gcs.Client.from_service_account_json(credentials_path)
        else:
            self.client = gcs.Client()

        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name

        # Verify bucket exists and is accessible
        if not self.bucket.exists():
            raise ValueError(
                f"Bucket '{bucket_name}' does not exist or is not accessible. "
                f"Create it in the GCS console first."
            )
        print(f"[GCS] Connected to bucket: {bucket_name}")

    def write_json(self, path: str, data: dict) -> None:
        """Upload JSON to GCS. Path becomes the object key (blob name)."""
        blob = self.bucket.blob(path)
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        blob.upload_from_string(json_str, content_type="application/json")

    def read_json(self, path: str) -> Optional[dict]:
        """Download and parse JSON from GCS."""
        blob = self.bucket.blob(path)
        if not blob.exists():
            return None
        content = blob.download_as_text()
        return json.loads(content)

    def file_exists(self, path: str) -> bool:
        """Check if a blob exists in the bucket."""
        return self.bucket.blob(path).exists()

    def list_files(self, prefix: str, suffix: str = "") -> List[str]:
        """
        List blobs under a prefix.
        GCS doesn't have real directories — everything is a flat key.
        Prefix acts like a directory filter.

        Example: list_files("manifests/", ".json")
        Returns: ["manifests/scrape_2025-01-15.json", ...]
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        results = []
        for blob in blobs:
            if not suffix or blob.name.endswith(suffix):
                results.append(blob.name)
        return sorted(results, reverse=True)