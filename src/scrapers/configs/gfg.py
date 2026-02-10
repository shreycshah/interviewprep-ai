from datetime import date

class GFGScraperConfigs:
    """Configuration for GFG Scraper"""

    SITEMAP_INDEX_URL = "https://www.geeksforgeeks.org/sitemap_index_new.xml"
    POST_SITEMAP_PREFIX = "https://www.geeksforgeeks.org/post/"
    INTERVIEW_URL_PATTERN = r"^https://www\.geeksforgeeks\.org/interview-experiences/[^/]*interview[^/]*"

    SCRAPE_TYPE = "bulk"
    BULK_START_DATE = "2025-01-01"
    DELAY_SECONDS = 3
    USER_AGENT = "InterviewPrepBot/1.0 (Academic Project; Non-commercial)"

    # Relative paths within storage (no more absolute Path references)
    RAW_PREFIX = "raw"
    # MANIFESTS_PREFIX = "manifests"

    @classmethod
    def get_today_str(cls) -> str:
        return date.today().isoformat()

    @classmethod
    def get_batch_id(cls, scrape_type: str) -> str:
        suffix = "bulk" if scrape_type == "bulk" else "weekly"
        return f"{cls.get_today_str()}_{suffix}"

    @classmethod
    def get_raw_prefix(cls, scrape_type: str) -> str:
        """
        Returns the raw prefix based on scrape type:
        - bulk: raw/bulk/gfg
        - incremental: raw/incremental/{date}/gfg
        """
        if scrape_type == "bulk":
            return f"{cls.RAW_PREFIX}/bulk/gfg"
        else:
            return f"{cls.RAW_PREFIX}/incremental/{cls.get_today_str()}/gfg"