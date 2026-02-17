from playwright.sync_api import sync_playwright
import xml.etree.ElementTree as ET
import os
import time
import json
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from datetime import date, datetime

from src.storage.storage_backend import StorageBackend
from src.scrapers.configs.medium import MediumScraperConfigs
from src.data_models.scraped_document import ScrapedInterviewDocument
from src.data_models.scraping_manifest import Manifest


"""Utility functions for web scraping."""

import hashlib
import random
import time

def random_delay(min_seconds, max_seconds):
    """Sleep for a random amount of time to mimic human behavior."""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def categorize_error(error_type):
    """Helper to categorize error types."""
    if "410" in error_type:
        return "http_410_gone", "410 GONE - Article deleted or account suspended"
    elif "404" in error_type:
        return "http_404_not_found", "404 NOT FOUND - Article doesn't exist"
    elif "403" in error_type:
        return "http_403_forbidden", "403 FORBIDDEN - Access denied"
    elif "429" in error_type:
        return "http_429_rate_limited", "429 RATE LIMITED - Too many requests"
    elif "TIMEOUT" in error_type:
        return "timeout", "TIMEOUT - Request took too long"
    elif "CONNECTION" in error_type:
        return "connection_error", "CONNECTION ERROR - Network issue"
    elif "SERVER_ERROR" in error_type or any(str(c) in error_type for c in range(500, 600)):
        return "http_5xx_server_error", f"SERVER ERROR - {error_type}"
    else:
        return "other", f"FETCH ERROR - {error_type}"


@dataclass
class SitemapInfo:
    url: str
    lastmod: Optional[str]

    def lastmod_date(self) -> Optional[date]:
        if not self.lastmod:
            return None
        try:
            date_str = self.lastmod.split("T")[0]
            return date.fromisoformat(date_str)
        except ValueError:
            return None


class MediumScraper:
    """Medium article scraper."""

    def __init__(
        self,
        scrape_type: str = None,
        config: MediumScraperConfigs = None,
        storage: StorageBackend = None,
        log_dir: str = None,
    ):
        self.config = config or MediumScraperConfigs()
        self.scrape_type = scrape_type or self.config.SCRAPE_TYPE
        self.batch_id = self.config.get_batch_id(self.scrape_type)

        if storage is None:
            raise ValueError(
                "A valid StorageBackend instance must be provided to MediumScraper. "
                "Got storage=None."
            )
        self.storage = storage

        # Build relative paths for this run
        self.today_raw_prefix = self.config.get_raw_prefix(self.batch_id)
        self.manifests_prefix = self.config.MANIFESTS_PREFIX

        # Local log file
        self.log_dir = Path(log_dir or "./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"medium_scraper_{self.batch_id}.log"
        self.logger = self._setup_logging()

        self.stats = {
            "total": 0,
            "success": 0,
            "paywalled": 0,
            "errors": {
                "http_410_gone": 0,
                "http_404_not_found": 0,
                "http_403_forbidden": 0,
                "http_429_rate_limited": 0,
                "http_5xx_server_error": 0,
                "timeout": 0,
                "connection_error": 0,
                "parse_error": 0,
                "other": 0
            }
        }

    def _setup_logging(self):
        """Setup logging with both console and local file output."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers on re-init
        logger.handlers.clear()

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler — full verbosity
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler — reduced verbosity
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info(f"Log file: {self.log_file}")
        return logger

    # ========================================================================
    # SITEMAP METHODS
    # ========================================================================

    def fetch_sitemap(self, url):
        """Fetch sitemap using Playwright."""
        download_path = None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self.config.USER_AGENT,
                    accept_downloads=True
                )
                page = context.new_page()

                download = None
                try:
                    with page.expect_download(timeout=self.config.DOWNLOAD_TIMEOUT) as download_info:
                        try:
                            page.goto(url, timeout=self.config.PAGE_TIMEOUT)
                        except Exception as e:
                            if "Download is starting" not in str(e):
                                raise
                    download = download_info.value
                except Exception:
                    pass

                if download:
                    download_path = f"temp_sitemap_{int(time.time())}.xml"
                    download.save_as(download_path)
                else:
                    content = page.content()
                    if content.startswith('<?xml') or '<urlset' in content or '<sitemapindex' in content:
                        download_path = f"temp_sitemap_{int(time.time())}.xml"
                        with open(download_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    else:
                        browser.close()
                        return None

                browser.close()

                if download_path and os.path.exists(download_path):
                    with open(download_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    os.remove(download_path)
                    self.logger.info(f"Successfully fetched sitemap: {url}")
                    return content
                return None

        except Exception as e:
            self.logger.error(f"Failed to fetch sitemap {url}: {str(e)}")
            if download_path and os.path.exists(download_path):
                os.remove(download_path)
            return None

    def filter_2025_sitemaps(self, xml_content):
        """Filter sitemaps based on config pattern."""
        root = ET.fromstring(xml_content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        filtered = []
        for sitemap in root.findall('.//ns:sitemap', ns):
            loc = sitemap.find('ns:loc', ns)
            lastmod = sitemap.find('ns:lastmod', ns)
            if loc is not None:
                url = loc.text
                if self.config.SITEMAP_FILTER_PATTERN in url:
                    filtered.append(SitemapInfo(
                        url=url,
                        lastmod=lastmod.text.strip() if lastmod is not None and lastmod.text else None
                    ))

        # Filter by BULK_START_DATE if in bulk mode
        if self.scrape_type == "bulk":
            cutoff = date.fromisoformat(self.config.BULK_START_DATE)
            filtered = [s for s in filtered if s.lastmod_date() is None or s.lastmod_date() >= cutoff]

        self.logger.info(f"Filtered {len(filtered)} sitemaps containing matching posts")
        return filtered

    def extract_article_urls(self, xml_content):
        """Extract article URLs from sitemap."""
        root = ET.fromstring(xml_content)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        urls = []
        for url_elem in root.findall('.//ns:url', ns):
            loc = url_elem.find('ns:loc', ns)
            if loc is not None:
                urls.append(loc.text)

        return urls

    def filter_interview_urls(self, urls):
        """Filter URLs containing interview keyword from config."""
        interview_urls = []
        for url in urls:
            if self.config.INTERVIEW_URL_KEYWORD in url.lower():
                interview_urls.append(url)

        if interview_urls:
            self.logger.info(f"Found {len(interview_urls)} interview URLs in sitemap")
        return interview_urls

    # ========================================================================
    # SCRAPING METHODS
    # ========================================================================

    def fetch_html(self, url):
        """Fetch HTML using Playwright to avoid bot detection."""
        try:
            random_delay(self.config.FETCH_DELAY_MIN, self.config.FETCH_DELAY_MAX)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self.config.USER_AGENT,
                    viewport={'width': self.config.VIEWPORT_WIDTH, 'height': self.config.VIEWPORT_HEIGHT},
                    locale=self.config.LOCALE,
                    timezone_id=self.config.TIMEZONE_ID,
                )

                page = context.new_page()
                response = page.goto(url, wait_until='domcontentloaded', timeout=self.config.PAGE_TIMEOUT)

                if response.status >= 400:
                    browser.close()

                    if response.status == 404:
                        return None, "HTTP_404_NOT_FOUND"
                    elif response.status == 410:
                        return None, "HTTP_410_GONE (Deleted or account suspended)"
                    elif response.status == 403:
                        return None, "HTTP_403_FORBIDDEN (Access denied/rate limited)"
                    elif response.status == 429:
                        self.logger.warning(f"Rate limited! Sleeping for {self.config.RATE_LIMIT_SLEEP} seconds...")
                        time.sleep(self.config.RATE_LIMIT_SLEEP)
                        return None, "HTTP_429_TOO_MANY_REQUESTS (Rate limited)"
                    elif response.status >= 500:
                        return None, f"HTTP_{response.status}_SERVER_ERROR"
                    else:
                        return None, f"HTTP_{response.status}_ERROR"

                random_delay(self.config.POST_LOAD_DELAY_MIN, self.config.POST_LOAD_DELAY_MAX)
                html = page.content()
                browser.close()

                self.logger.info(f"Successfully fetched with Playwright: {url}")
                return html, None

        except Exception as e:
            error_type = str(e)

            if "Timeout" in error_type or "timeout" in error_type:
                self.logger.warning(f"Timeout for {url}")
                return None, "TIMEOUT (Request took too long)"
            elif "net::ERR" in error_type:
                self.logger.warning(f"Connection error for {url}")
                return None, "CONNECTION_ERROR (Network issue)"
            else:
                self.logger.error(f"Playwright error for {url}: {str(e)}")
                return None, f"PLAYWRIGHT_ERROR: {str(e)}"

    def is_paywalled(self, soup):
        """Robust paywall detection with enhanced Apollo State checking."""

        if soup.find("article", {"class": re.compile(r".*meteredContent.*", re.I)}):
            return True, "meteredContent class"

        if soup.find(string=re.compile(r"Member-only story", re.I)):
            return True, "Member-only badge"

        json_ld = soup.find("script", {"type": "application/ld+json"})
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if data.get("isAccessibleForFree") == False:
                    return True, "JSON-LD isAccessibleForFree"
            except (json.JSONDecodeError, Exception):
                pass

        apollo_state = soup.find("script", string=re.compile(r"__APOLLO_STATE__", re.I))
        if apollo_state:
            try:
                json_str = apollo_state.string.split("=", 1)[1].strip()
                if json_str.endswith(";"):
                    json_str = json_str[:-1]

                data = json.loads(json_str)

                for key, value in data.items():
                    if key.startswith("Post:") and isinstance(value, dict):
                        if value.get("isLocked") == True:
                            return True, "Apollo isLocked"
                        if value.get("isLockedPreviewOnly") == True:
                            return True, "Apollo isLockedPreviewOnly"
                        if value.get("isMarkedPaywallOnly") == True:
                            return True, "Apollo isMarkedPaywallOnly"

                        content_key = 'content({"postMeteringOptions":{"referrer":""}})'
                        content_ref = value.get(content_key) or value.get("content")
                        if isinstance(content_ref, dict):
                            if content_ref.get("isLockedPreviewOnly") == True:
                                return True, "Apollo content isLockedPreviewOnly"
            except (json.JSONDecodeError, Exception):
                pass

        if soup.find(string=re.compile(r"This is a preview", re.I)):
            return True, "Preview text"

        if soup.find(string=re.compile(r"Upgrade to continue|Subscribe to read", re.I)):
            return True, "Paywall CTA"

        return False, None

    def extract_metadata(self, soup, url):
        """Extract all available metadata from Medium article."""
        metadata = {}

        desc_meta = soup.find("meta", property="og:description") or soup.find("meta", {"name": "description"})
        if desc_meta:
            metadata["description"] = desc_meta.get("content", "")

        read_time_meta = soup.find("meta", {"name": "twitter:data1"})
        if read_time_meta:
            metadata["reading_time"] = read_time_meta.get("content", "")

        tags = []
        apollo_script = soup.find("script", string=re.compile(r"__APOLLO_STATE__", re.I))
        if apollo_script:
            try:
                json_str = apollo_script.string.split("=", 1)[1].strip()
                if json_str.endswith(";"):
                    json_str = json_str[:-1]

                data = json.loads(json_str)

                for key, value in data.items():
                    if key.startswith("Tag:") and isinstance(value, dict):
                        tag_title = value.get("displayTitle") or value.get("id")
                        if tag_title and tag_title not in tags:
                            tags.append(tag_title)

                if tags:
                    metadata["tags"] = tags
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"  Failed to extract tags from Apollo state: {str(e)}")

        image_meta = soup.find("meta", property="og:image")
        if image_meta:
            metadata["featured_image"] = image_meta.get("content", "")

        canonical = soup.find("link", rel="canonical")
        if canonical:
            metadata["canonical_url"] = canonical.get("href", "")

        return metadata

    def extract_from_apollo_state(self, soup):
        """Extract article content from Medium's Apollo GraphQL state."""
        try:
            apollo_script = soup.find("script", string=re.compile(r"__APOLLO_STATE__", re.I))

            if not apollo_script:
                return None

            json_str = apollo_script.string.split("=", 1)[1].strip()
            if json_str.endswith(";"):
                json_str = json_str[:-1]

            data = json.loads(json_str)

            paragraphs = []
            for key, value in data.items():
                if key.startswith("Paragraph:") and isinstance(value, dict):
                    text = value.get("text", "")
                    paragraph_type = value.get("type", "P")

                    if text.strip():
                        try:
                            index = int(key.split("_")[-1])
                        except (ValueError, IndexError):
                            index = 0

                        paragraphs.append({
                            "index": index,
                            "text": text,
                            "type": paragraph_type
                        })

            if not paragraphs:
                return None

            paragraphs.sort(key=lambda x: x["index"])

            markdown_content = []
            for para in paragraphs:
                text = para["text"]
                para_type = para["type"]

                if para_type == "H2":
                    markdown_content.append(f"## {text}")
                elif para_type == "H3":
                    markdown_content.append(f"### {text}")
                elif para_type == "H4":
                    markdown_content.append(f"#### {text}")
                elif para_type == "PQ":
                    markdown_content.append(f"> {text}")
                elif para_type == "PRE":
                    markdown_content.append(f"```\n{text}\n```")
                elif para_type in ["OLI", "ULI"]:
                    markdown_content.append(f"- {text}")
                else:
                    markdown_content.append(text)

            content = "\n\n".join(markdown_content)

            self.logger.info(f"Extracted {len(paragraphs)} paragraphs from Apollo state")
            return content

        except Exception as e:
            self.logger.warning(f"Failed to extract from Apollo state: {str(e)}")
            return None

    def parse_article(self, html, url):
        """Parse Medium article with Apollo State + HTML fallback."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Check for paywall FIRST
            is_paywalled_result, paywall_reason = self.is_paywalled(soup)
            if is_paywalled_result:
                self.logger.info(f"Paywalled article skipped: {url} - {paywall_reason}")
                return None, f"PAYWALLED ({paywall_reason})"

            # Extract title
            title_meta = soup.find("meta", property="og:title")
            title = title_meta["content"] if title_meta else "Untitled"

            # Extract published date
            date_meta = soup.find("meta", property="article:published_time")
            published_at = date_meta["content"] if date_meta else None

            # Try Apollo State FIRST
            raw_content = self.extract_from_apollo_state(soup)
            extraction_method = "Apollo State"

            if not raw_content or len(raw_content) < self.config.MIN_CONTENT_LENGTH:
                self.logger.info(f"Apollo extraction insufficient, trying HTML strategies")

                article_tag = None

                article_tag = soup.find("article")
                if article_tag:
                    extraction_method = "Article tag"

                if not article_tag:
                    article_tag = soup.find("div", {"class": re.compile(r".*postArticle.*", re.I)})
                    if article_tag:
                        extraction_method = "postArticle class"

                if not article_tag:
                    article_tag = soup.find("main") or soup.find("div", {"role": "main"})
                    if article_tag:
                        extraction_method = "Main tag"

                if not article_tag:
                    article_tag = soup.find("div", {"data-testid": re.compile(r".*content.*|.*article.*", re.I)})
                    if article_tag:
                        extraction_method = "data-testid"

                if not article_tag:
                    all_paragraphs = soup.find_all("p")
                    meaningful_paragraphs = [p for p in all_paragraphs
                                            if len(p.get_text(strip=True)) > self.config.MIN_MEANINGFUL_PARAGRAPH_LENGTH]

                    if len(meaningful_paragraphs) >= self.config.MIN_MEANINGFUL_PARAGRAPH_COUNT:
                        article_tag = soup.new_tag("div")
                        for p in meaningful_paragraphs:
                            article_tag.append(p)
                        extraction_method = f"Paragraph extraction ({len(meaningful_paragraphs)} paras)"
                        self.logger.info(extraction_method)
                    else:
                        self.logger.warning(f"No content found - DROPPING")
                        self.logger.warning(f"  Title: {title}")
                        self.logger.warning(f"  Total <p> tags: {len(all_paragraphs)}")
                        self.logger.warning(f"  Meaningful paragraphs: {len(meaningful_paragraphs)}")
                        return None, "NO_CONTENT"

                raw_content = md(str(article_tag)).strip()

            # Validate content length
            if len(raw_content) < self.config.MIN_CONTENT_LENGTH:
                self.logger.warning(f"Content too short: {len(raw_content)} chars - DROPPING")
                self.logger.warning(f"  Title: {title}")
                self.logger.warning(f"  Method: {extraction_method}")
                return None, "CONTENT_TOO_SHORT"

            # Extract metadata
            source_metadata = self.extract_metadata(soup, url)

            # Build ScrapedInterviewDocument
            doc = ScrapedInterviewDocument(
                document_id=ScrapedInterviewDocument.generate_document_id("medium", url),
                source_platform="medium",
                source_url=url,
                title=title,
                raw_content=raw_content,
                published_at=published_at,
                scraped_at=ScrapedInterviewDocument.now_iso(),
                scrape_type=self.scrape_type,
                scrape_batch_id=self.batch_id,
                source_metadata=source_metadata,
            )

            self.logger.info(f"Successfully parsed: {title}")
            self.logger.info(f"   Method: {extraction_method}, Length: {len(raw_content)} chars")

            return doc, None

        except Exception as e:
            self.logger.error(f"Parse error for {url}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, f"PARSE_ERROR: {str(e)}"

    def _save_document(self, doc: ScrapedInterviewDocument):
        """Save document via storage backend — works for both local and GCS."""
        path = f"{self.today_raw_prefix}/{doc.document_id}.json"
        self.storage.write_json(path, doc.to_dict())

    def scrape_article(self, url):
        """Scrape a single article."""

        # Fetch HTML
        html, fetch_error = self.fetch_html(url)

        if fetch_error:
            return {
                "url": url,
                "status": "FETCH_ERROR",
                "error_type": fetch_error,
            }

        # Parse article
        doc, parse_error = self.parse_article(html, url)

        if parse_error:
            return {
                "url": url,
                "status": "PARSE_ERROR",
                "error_type": parse_error,
            }

        # Save via storage backend
        self._save_document(doc)

        return {
            "url": url,
            "status": "SUCCESS",
            "document_id": doc.document_id,
        }

    # ========================================================================
    # WORKFLOW METHODS
    # ========================================================================

    def collect_interview_urls(self, sitemap_urls):
        """Collect all interview URLs from sitemaps."""
        self.logger.info("="*60)
        self.logger.info("PHASE 1: COLLECTING INTERVIEW URLS")
        self.logger.info("="*60)

        all_interview_urls = []

        for i, sitemap in enumerate(sitemap_urls, 1):
            self.logger.info(f"Processing sitemap {i}/{len(sitemap_urls)}: {sitemap.url}")

            sitemap_content = self.fetch_sitemap(sitemap.url)
            if sitemap_content is None:
                continue

            article_urls = self.extract_article_urls(sitemap_content)
            interview_urls = self.filter_interview_urls(article_urls)
            all_interview_urls.extend(interview_urls)

            time.sleep(1)

        self.logger.info(f"Total interview URLs collected: {len(all_interview_urls)}")
        return all_interview_urls

    def scrape_all_articles(self, urls):
        """Scrape all articles and update stats."""
        self.logger.info("="*60)
        self.logger.info("PHASE 2: SCRAPING ARTICLES")
        self.logger.info("="*60)
        self.logger.info(f"Using random delays ({self.config.FETCH_DELAY_MIN}-{self.config.FETCH_DELAY_MAX}s fetch, "
                         f"{self.config.BETWEEN_ARTICLES_DELAY_MIN}-{self.config.BETWEEN_ARTICLES_DELAY_MAX}s between articles)")

        for idx, url in enumerate(urls, 1):
            self.logger.info(f"Scraping {idx}/{len(urls)}: {url}")

            result = self.scrape_article(url)
            self._update_stats(result)

            random_delay(self.config.BETWEEN_ARTICLES_DELAY_MIN, self.config.BETWEEN_ARTICLES_DELAY_MAX)

    def _update_stats(self, result):
        """Update statistics based on scrape result."""
        if result.get("status") == "SUCCESS":
            self.stats["success"] += 1

        elif result.get("status") == "PARSE_ERROR":
            error_type = result.get("error_type", "")

            if "PAYWALLED" in error_type:
                self.stats["paywalled"] += 1
            else:
                self.stats["errors"]["parse_error"] += 1

        elif result.get("status") == "FETCH_ERROR":
            error_type = result.get("error_type", "")
            category, _ = categorize_error(error_type)
            self.stats["errors"][category] += 1

        else:
            self.stats["errors"]["other"] += 1

    def print_summary(self):
        """Print final statistics."""
        print("\n" + "=" * 60)
        print("SCRAPE COMPLETE")
        print("=" * 60)
        print(f"Files collected: {self.stats['success']}")
        print(f"Paywalled (skipped): {self.stats['paywalled']}")
        print(f"Errors: {self.stats['errors']}")

        if self.stats['errors']['http_403_forbidden'] > 0:
            print(f"\nWARNING: {self.stats['errors']['http_403_forbidden']} articles blocked (403)")

        print(f"\nFull log: {self.log_file}")
        print("=" * 60)

    # ─────────── Manifest ───────────
    def _create_manifest(self, started_at: str) -> Manifest:
        total_errors = sum(self.stats["errors"].values())

        manifest = Manifest(
            scrape_date=self.config.get_today_str(),
            scrape_type=self.scrape_type,
            started_at=started_at,
            completed_at=ScrapedInterviewDocument.now_iso(),
            sources={
                "medium": {
                    "files_collected": self.stats["success"],
                    "paywalled_skipped": self.stats["paywalled"],
                    "total_urls": self.stats["total"],
                    "errors": self.stats["errors"],
                    "total_errors": total_errors,
                }
            },
            total_files=self.stats["success"],
        )

        manifest_path = f"{self.manifests_prefix}/scrape_{self.config.get_today_str()}.json"
        manifest.save(self.storage, manifest_path)
        print(f"\nManifest saved to {manifest_path}")
        return manifest

    # ========================================================================
    # MAIN RUN METHOD
    # ========================================================================

    def run(self):
        """Main pipeline execution."""
        print("=" * 60)
        print(f"Medium Interview Scraper - {self.scrape_type.upper()} mode")
        print(f"Storage: {self.storage.__class__.__name__}")
        print("=" * 60)
        print(f"Batch ID: {self.batch_id}")

        start_time = datetime.now()
        started_at = start_time.isoformat() + "Z"

        # Fetch main sitemap
        self.logger.info("Fetching main sitemap...")
        xml_content = self.fetch_sitemap(self.config.SITEMAP_INDEX_URL)
        if xml_content is None:
            print("ERROR: Failed to fetch main sitemap. Exiting.")
            return

        # Filter sitemaps
        sitemap_urls = self.filter_2025_sitemaps(xml_content)

        # Apply max sitemaps limit (for testing)
        if self.config.MAX_SITEMAPS is not None:
            sitemap_urls = sitemap_urls[:self.config.MAX_SITEMAPS]
            self.logger.info(f"Limited to {self.config.MAX_SITEMAPS} sitemaps")

        # Collect interview URLs
        interview_urls = self.collect_interview_urls(sitemap_urls)
        self.stats["total"] = len(interview_urls)

        if not interview_urls:
            print("No interview URLs found. Exiting.")
            return

        # Scrape all articles
        self.scrape_all_articles(interview_urls)

        # Save manifest
        manifest = self._create_manifest(started_at)

        # Print summary
        end_time = datetime.now()
        delta = end_time - start_time
        total_seconds = int(delta.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)

        self.print_summary()
        print(f"Total time taken for Medium Scraping: {minutes} min {seconds} sec")



# ============== ENTRY POINT ==============
# from src.storage.gcs_backend import GCSBackend
#
# if __name__ == "__main__":
#     storage = GCSBackend(
#         bucket_name="interviewprep-ai-data",
#         project_id="professorbot-dovbsg",
#         secret_name="gcs-service-account-key",
#     )
#
#     config = MediumScraperConfigs()
#
#     scraper = MediumScraper(
#         scrape_type=config.SCRAPE_TYPE,
#         config=config,
#         storage=storage,
#     )
#     scraper.run()