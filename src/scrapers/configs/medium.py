from datetime import date


class MediumScraperConfigs:
    """Configuration for Medium Scraper"""

    # Sitemap
    SITEMAP_INDEX_URL = "https://medium.com/sitemap/sitemap.xml"
    SITEMAP_FILTER_PATTERN = "posts/2025/posts-2025"

    # URL filtering
    INTERVIEW_URL_KEYWORD = "interview-experience"

    # Scrape settings
    SCRAPE_TYPE = "bulk"
    BULK_START_DATE = "2025-01-01"

    # Delays (seconds)
    FETCH_DELAY_MIN = 2
    FETCH_DELAY_MAX = 5
    BETWEEN_ARTICLES_DELAY_MIN = 3
    BETWEEN_ARTICLES_DELAY_MAX = 8
    POST_LOAD_DELAY_MIN = 1
    POST_LOAD_DELAY_MAX = 3
    RATE_LIMIT_SLEEP = 30

    # Browser settings
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    VIEWPORT_WIDTH = 1920
    VIEWPORT_HEIGHT = 1080
    LOCALE = "en-US"
    TIMEZONE_ID = "America/New_York"

    # Content validation
    MIN_CONTENT_LENGTH = 100
    MIN_MEANINGFUL_PARAGRAPH_LENGTH = 15
    MIN_MEANINGFUL_PARAGRAPH_COUNT = 2

    # Timeouts (ms)
    PAGE_TIMEOUT = 30000
    DOWNLOAD_TIMEOUT = 15000

    # Storage paths (relative within storage)
    RAW_PREFIX = "raw"
    MANIFESTS_PREFIX = "manifests/medium"

    # Test mode
    MAX_SITEMAPS = None  # Set to None to process all

    @classmethod
    def get_today_str(cls) -> str:
        return date.today().isoformat()

    @classmethod
    def get_batch_id(cls, scrape_type: str) -> str:
        suffix = "bulk" if scrape_type == "bulk" else "weekly"
        return f"{cls.get_today_str()}_{suffix}"

    @classmethod
    def get_raw_prefix(cls, batch_id: str) -> str:
        return f"{cls.RAW_PREFIX}/{batch_id}/medium"