import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import time
import re
from datetime import date, datetime
from dataclasses import dataclass
from typing import Optional, List, Dict

from src.storage.storage_backend import StorageBackend
from src.storage.gcs_backend import GCSBackend
from src.scrapers.configs.gfg import GFGScraperConfigs
from src.data_models.scraped_document import ScrapedInterviewDocument
from src.data_models.scraping_manifest import Manifest


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


# ============== GFG SCRAPER CLASS ==============
class GFGScraper:
    """
    GeeksforGeeks Interview Experience Scraper
    Now accepts a StorageBackend for flexible output (local or GCS).
    """

    def __init__(
        self,
        scrape_type: str = None,
        config: GFGScraperConfigs = None,
        storage: StorageBackend = None,
    ):
        self.config = config or GFGScraperConfigs()
        self.scrape_type = scrape_type or self.config.SCRAPE_TYPE
        self.batch_id = self.config.get_batch_id(self.scrape_type)

        if storage is None:
            raise ValueError(
                "A valid StorageBackend instance must be provided to GFGScraper. "
                "Got storage=None."
            )
        self.storage = storage

        # Build relative paths for this run
        self.today_raw_prefix = self.config.get_raw_prefix(self.batch_id)
        self.manifests_prefix = self.config.MANIFESTS_PREFIX

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        self.stats = {
            "files_collected": 0,
            "sitemaps_processed": 0,
            "urls_found": 0,
            "errors": 0,
            "error_urls": [],
        }

    # ─────────── HTTP Methods ───────────
    def _fetch(self, url: str, delay: bool = True) -> Optional[str]:
        try:
            if delay:
                time.sleep(self.config.DELAY_SECONDS)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"    ERROR fetching {url}: {e}")
            self.stats["errors"] += 1
            self.stats["error_urls"].append(url)
            return None

    def _fetch_xml(self, url: str) -> Optional[ET.Element]:
        content = self._fetch(url)
        if content:
            try:
                return ET.fromstring(content)
            except ET.ParseError as e:
                print(f"    ERROR parsing XML: {e}")
        return None

    # ─────────── Sitemap Processing ───────────
    def _get_sitemaps(self) -> List[SitemapInfo]:
        print("\n[1/3] Fetching sitemap index...")
        root = self._fetch_xml(self.config.SITEMAP_INDEX_URL)
        if root is None:
            return []

        sitemaps = []
        ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        for sitemap_elem in root.findall(".//ns:sitemap", ns):
            loc = sitemap_elem.find("ns:loc", ns)
            lastmod = sitemap_elem.find("ns:lastmod", ns)
            if loc is not None and loc.text:
                url = loc.text.strip()
                if url.startswith(self.config.POST_SITEMAP_PREFIX):
                    sitemaps.append(SitemapInfo(
                        url=url,
                        lastmod=lastmod.text.strip() if lastmod is not None and lastmod.text else None,
                    ))

        if not sitemaps:
            for sitemap_elem in root.iter():
                if sitemap_elem.tag.endswith("sitemap"):
                    loc = lastmod = None
                    for child in sitemap_elem:
                        if child.tag.endswith("loc") and child.text:
                            loc = child.text.strip()
                        if child.tag.endswith("lastmod") and child.text:
                            lastmod = child.text.strip()
                    if loc and loc.startswith(self.config.POST_SITEMAP_PREFIX):
                        sitemaps.append(SitemapInfo(url=loc, lastmod=lastmod))

        sitemaps.sort(key=lambda x: x.lastmod if x.lastmod else "0000-00-00", reverse=True)
        print(f"    Found {len(sitemaps)} post sitemaps")
        return sitemaps

    def _filter_sitemaps(self, sitemaps: List[SitemapInfo]) -> List[SitemapInfo]:
        if self.scrape_type == "bulk":
            cutoff = date.fromisoformat(self.config.BULK_START_DATE)
            filtered = [s for s in sitemaps if s.lastmod_date() is None or s.lastmod_date() >= cutoff]
            print(f"    Filtered to {len(filtered)} sitemaps (lastmod >= {cutoff})")
            return filtered
        else:
            last_manifest = Manifest.get_latest(self.storage, self.manifests_prefix)
            if last_manifest and last_manifest.last_sitemap_lastmod:
                cutoff_str = last_manifest.last_sitemap_lastmod
                filtered = [s for s in sitemaps if s.lastmod and s.lastmod > cutoff_str]
                print(f"    Incremental: {len(filtered)} sitemaps newer than {cutoff_str}")
                return filtered
            else:
                print("    No previous manifest found, processing all sitemaps")
                return sitemaps

    def _extract_interview_urls(self, sitemaps: List[SitemapInfo]) -> List[Dict]:
        print(f"\n[2/3] Extracting interview URLs from {len(sitemaps)} sitemaps...")
        pattern = re.compile(self.config.INTERVIEW_URL_PATTERN)
        all_urls = []

        if self.scrape_type == "bulk":
            cutoff_date = self.config.BULK_START_DATE
        else:
            last_manifest = Manifest.get_latest(self.storage, self.manifests_prefix)
            if last_manifest and last_manifest.last_sitemap_lastmod:
                cutoff_date = last_manifest.last_sitemap_lastmod.split("T")[0]
            else:
                cutoff_date = None

        print(f"    URL lastmod cutoff: {cutoff_date or 'None (all URLs)'}")

        for i, sitemap in enumerate(sitemaps):
            print(f"    [{i+1}/{len(sitemaps)}] Processing {sitemap.url[:60]}...")
            root = self._fetch_xml(sitemap.url)
            if root is None:
                continue

            urls_in_sitemap = []
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            for url_elem in root.findall(".//ns:url", ns):
                loc = url_elem.find("ns:loc", ns)
                lastmod = url_elem.find("ns:lastmod", ns)
                if loc is not None and loc.text:
                    urls_in_sitemap.append({
                        "url": loc.text.strip(),
                        "lastmod": lastmod.text.strip() if lastmod is not None and lastmod.text else None,
                    })

            if not urls_in_sitemap:
                for url_elem in root.iter():
                    if url_elem.tag.endswith("url"):
                        loc = lastmod = None
                        for child in url_elem:
                            if child.tag.endswith("loc") and child.text:
                                loc = child.text.strip()
                            if child.tag.endswith("lastmod") and child.text:
                                lastmod = child.text.strip()
                        if loc:
                            urls_in_sitemap.append({"url": loc, "lastmod": lastmod})

            interview_urls = [u for u in urls_in_sitemap if pattern.match(u["url"])]

            if cutoff_date:
                filtered_urls = []
                for u in interview_urls:
                    if u["lastmod"]:
                        if u["lastmod"].split("T")[0] >= cutoff_date:
                            filtered_urls.append(u)
                    elif self.scrape_type == "bulk":
                        filtered_urls.append(u)
                interview_urls = filtered_urls

            all_urls.extend(interview_urls)
            print(f"        Found {len(interview_urls)} interview URLs (after lastmod filter)")
            self.stats["sitemaps_processed"] += 1
            # break  # Testing for single site map

        seen = set()
        unique_urls = []
        for u in all_urls:
            if u["url"] not in seen:
                seen.add(u["url"])
                unique_urls.append(u)

        self.stats["urls_found"] = len(unique_urls)
        print(f"    Total unique interview URLs: {len(unique_urls)}")
        return unique_urls

    # ─────────── Article Parsing ───────────
    def _parse_article(self, url: str, html: str) -> Optional[ScrapedInterviewDocument]:
        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "Untitled"

        article = soup.find("div", class_=lambda c: c and "article--viewer" in c)

        if article:
            for unwanted in article.find_all("div", class_=lambda c: c and "LeftbarOutsideIndiaContent" in str(c)):
                unwanted.decompose()
            for tag in article.find_all(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            raw_content = article.get_text(separator="\n", strip=True)
        else:
            body = soup.find("body")
            if body:
                for tag in body.find_all(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                raw_content = body.get_text(separator="\n", strip=True)
            else:
                raw_content = ""

        if not raw_content:
            return None

        published_at = None
        date_div = soup.find("div", class_=lambda c: c and "ArticleHeader_last_updated_parent" in str(c))
        if date_div:
            date_text = date_div.get_text(strip=True)
            match = re.search(r"Last Updated\s*:\s*(.+)", date_text)
            if match:
                published_at = match.group(1).strip()

        tags = []
        tags_container = soup.find("div", class_=lambda c: c and "HeadingAndChipComponent_mainContainer__dataChips" in str(c))
        if tags_container:
            for tag_elem in tags_container.find_all("a"):
                tag_text = tag_elem.get_text(strip=True)
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)

        exp_type = "unknown"
        tags_lower = [t.lower() for t in tags]
        title_lower = title.lower()
        if "on-campus" in tags_lower or "on campus" in title_lower:
            exp_type = "on-campus"
        elif "off-campus" in tags_lower or "off campus" in title_lower:
            exp_type = "off-campus"
        elif "internship" in tags_lower or "intern" in title_lower:
            exp_type = "internship"

        return ScrapedInterviewDocument(
            document_id=ScrapedInterviewDocument.generate_document_id("gfg",url),
            source_platform="gfg",
            source_url=url,
            title=title,
            raw_content=raw_content,
            published_at=published_at,
            scraped_at=ScrapedInterviewDocument.now_iso(),
            scrape_type=self.scrape_type,
            scrape_batch_id=self.batch_id,
            source_metadata={"tags": tags, "experience_type": exp_type},
        )

    # ─────────── Scraping ───────────
    def _scrape_articles(self, urls: List[Dict]) -> List[ScrapedInterviewDocument]:
        print(f"\n[3/3] Scraping {len(urls)} articles...")
        documents = []

        for i, url_data in enumerate(urls):
            url = url_data["url"]
            print(f"    [{i+1}/{len(urls)}] {url[:65]}...")

            html = self._fetch(url)
            if html is None:
                continue

            doc = self._parse_article(url, html)
            if doc:
                self._save_document(doc)
                documents.append(doc)
                self.stats["files_collected"] += 1
            else:
                print(f"        WARNING: Could not parse content")
                self.stats["errors"] += 1
                self.stats["error_urls"].append(url)

        return documents

    def _save_document(self, doc: ScrapedInterviewDocument):
        """Save document via storage backend — works for both local and GCS."""
        path = f"{self.today_raw_prefix}/{doc.document_id}.json"
        self.storage.write_json(path, doc.to_dict())

    # ─────────── Manifest ───────────
    def _create_manifest(self, sitemaps: List[SitemapInfo], started_at: str) -> Manifest:
        latest_lastmod = None
        if sitemaps:
            latest_lastmod = max(
                (s.lastmod for s in sitemaps if s.lastmod), default=None
            )

        manifest = Manifest(
            scrape_date=self.config.get_today_str(),
            scrape_type=self.scrape_type,
            started_at=started_at,
            completed_at=ScrapedInterviewDocument.now_iso(),
            sources={
                "gfg": {
                    "files_collected": self.stats["files_collected"],
                    "sitemaps_processed": self.stats["sitemaps_processed"],
                    "urls_found": self.stats["urls_found"],
                    "errors": self.stats["errors"],
                    "error_urls": self.stats["error_urls"][:10],
                }
            },
            total_files=self.stats["files_collected"],
            last_sitemap_lastmod=latest_lastmod,
        )

        manifest_path = f"{self.manifests_prefix}/scrape_{self.config.get_today_str()}.json"
        manifest.save(self.storage, manifest_path)
        print(f"\nManifest saved to {manifest_path}")
        return manifest

    # ─────────── Main Entry Point ───────────
    def run(self):
        print("=" * 60)
        print(f"GFG Interview Scraper - {self.scrape_type.upper()} mode")
        print(f"Storage: {self.storage.__class__.__name__}")
        print("=" * 60)
        print(f"Batch ID: {self.batch_id}")

        started_at = ScrapedInterviewDocument.now_iso()
        start_time = datetime.now()

        sitemaps = self._get_sitemaps()
        if not sitemaps:
            print("ERROR: No sitemaps found. Exiting.")
            return

        sitemaps = self._filter_sitemaps(sitemaps)
        if not sitemaps:
            print("No sitemaps to process after filtering. Exiting.")
            return

        urls = self._extract_interview_urls(sitemaps)
        if not urls:
            print("No interview URLs found. Exiting.")
            return

        documents = self._scrape_articles(urls)
        manifest = self._create_manifest(sitemaps, started_at)

        end_time = datetime.now()
        delta = end_time - start_time
        total_seconds = int(delta.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)

        print("\n" + "=" * 60)
        print("SCRAPE COMPLETE")
        print("=" * 60)
        print(f"Files collected: {self.stats['files_collected']}")
        print(f"Errors: {self.stats['errors']}")

        print(f"Total time taken for GFG Scraping: {minutes} min {seconds} sec")

# ============== ENTRY POINT ==============
# if __name__ == "__main__":
#     storage = GCSBackend(
#         bucket_name="interviewprep-ai-data",
#         project_id="professorbot-dovbsg",
#         secret_name="gcs-service-account-key",
#     )
#     scraper = GFGScraper(scrape_type=GFGScraperConfigs().SCRAPE_TYPE, storage=storage)
#     scraper.run()