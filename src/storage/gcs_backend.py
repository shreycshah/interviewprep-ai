import json
import tempfile
from typing import Optional, List
from src.storage.storage_backend import StorageBackend
from google.cloud import storage as gcs
import os

def _load_credentials_from_secret_manager(
    project_id: str,
    secret_name: str,
    version: str = "latest",
) -> str:
    """
    Fetch the service account JSON key from Google Secret Manager
    and write it to a temporary file.

    Returns the path to the temp file (needed by the GCS client).

    NOTE: The caller (GCSBackend) must bootstrap authentication first.
    Typically this means the machine running the code has *some* way to
    reach Secret Manager — either via:
        * GOOGLE_APPLICATION_CREDENTIALS pointing to a minimal key
        * Default credentials on a GCE/Cloud Run/GKE environment
        * gcloud auth application-default login (local dev)
    """
    from google.cloud import secretmanager

    sm_client = secretmanager.SecretManagerServiceClient()

    # Build the resource name: projects/PROJECT/secrets/SECRET/versions/VERSION
    resource = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"

    response = sm_client.access_secret_version(request={"name": resource})
    secret_payload = response.payload.data.decode("UTF-8")

    # Write to a temp file because the GCS client expects a file path
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="gcs_sa_"
    )
    tmp.write(secret_payload)
    tmp.close()

    return tmp.name

class GCSBackend(StorageBackend):
    """
    Google Cloud Storage implementation of StorageBackend.

    Supports three authentication methods:
        1. Secret Manager: pass project_id + secret_name
        2. Local key file: pass credentials_path
        3. Environment default: set GOOGLE_APPLICATION_CREDENTIALS

    Usage:
        # Via Secret Manager (recommended for production/shared environments)
        storage = GCSBackend(
            bucket_name="interviewprep-ai-data",
            project_id="professorbot-dovbsg",
            secret_name="gcs-service-account-key",
        )

        # Via env var (CI/CD or when env is pre-configured)
        storage = GCSBackend(bucket_name="interviewprep-ai-data")
    """

    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        secret_name: Optional[str] = None,
    ):
        self._tmp_key_path: Optional[str] = None

        # ── Auth Priority ──
        # 1. Secret Manager (with fallback to ADC on failure)
        if project_id and secret_name:
            try:
                print(f"[GCS] Loading credentials from Secret Manager: {secret_name}")
                self._tmp_key_path = _load_credentials_from_secret_manager(
                    project_id=project_id,
                    secret_name=secret_name,
                )
                self.client = gcs.Client.from_service_account_json(self._tmp_key_path)
            except Exception as e:
                print(f"[GCS] Secret Manager failed ({e}), falling back to default credentials")
                self.client = gcs.Client()

        # 2. Default (env var or metadata server)
        else:
            print("[GCS] Using default credentials (env var or compute metadata)")
            self.client = gcs.Client()

        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name

        # Verify bucket is accessible
        if not self.bucket.exists():
            self._cleanup_temp_key()
            raise ValueError(
                f"Bucket '{bucket_name}' does not exist or is not accessible. "
                f"Create it in the GCS console first."
            )
        print(f"[GCS] Connected to bucket: {bucket_name}")

    def _cleanup_temp_key(self):
        """Remove the temporary key file created from Secret Manager."""
        if self._tmp_key_path and os.path.exists(self._tmp_key_path):
            os.unlink(self._tmp_key_path)
            self._tmp_key_path = None

    def __del__(self):
        """Cleanup temp key file when the object is garbage collected."""
        self._cleanup_temp_key()

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