"""
LeetCode Interview Experience Scraper (GCS-compatible)
=====================================================
Refactored from leetcode_scraper_final.py to match GFG scraper architecture.
Supports bulk and incremental scraping with manifest tracking.
Works with both local filesystem and Google Cloud Storage.

Usage:
    from src.storage.gcs_backend import GCSBackend

    storage = GCSBackend(bucket_name="interviewprep-ai-data")
    scraper = LeetCodeScraper(scrape_type="bulk", storage=storage)
    scraper.run()
"""

import requests
import json
import time
import re
import hashlib
from datetime import datetime, date, timezone
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from collections import Counter

from src.storage.storage_backend import StorageBackend


# ============== CONFIGURATION ==============
class ScraperConfig:
    """Configuration for LeetCode Scraper — mirrors GFG ScraperConfig."""

    GRAPHQL_URL = "https://leetcode.com/graphql"
    TARGET_TAGS = ["interview"]

    SCRAPE_TYPE = "bulk"
    BULK_MAX_POSTS = 500
    INCREMENTAL_MAX_POSTS = 100
    PAGE_SIZE = 50
    DELAY_SECONDS = 1.0
    DETAIL_DELAY_SECONDS = 1.0

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    RAW_PREFIX = "raw/leetcode"
    MANIFESTS_PREFIX = "manifests"

    # ── GraphQL Queries (captured from browser DevTools) ──

    LIST_QUERY = """
    query discussPostItems($orderBy: ArticleOrderByEnum, $keywords: [String]!, $tagSlugs: [String!], $skip: Int, $first: Int) {
      ugcArticleDiscussionArticles(
        orderBy: $orderBy
        keywords: $keywords
        tagSlugs: $tagSlugs
        skip: $skip
        first: $first
      ) {
        totalNum
        pageInfo { hasNextPage }
        edges {
          node {
            uuid
            title
            slug
            summary
            author {
              realName
              userAvatar
              userSlug
              userName
              nameColor
              certificationLevel
            }
            isAnonymous
            articleType
            thumbnail
            summary
            createdAt
            updatedAt
            topicId
            hitCount
            reactions { count reactionType }
            tags { name slug tagType }
            topic { id topLevelCommentCount }
          }
        }
      }
    }
    """

    DETAIL_QUERY = """
    query discussPostDetail($topicId: ID!) {
      ugcArticleDiscussionArticle(topicId: $topicId) {
        uuid
        title
        slug
        summary
        content
        author { userName userSlug }
        createdAt
        updatedAt
        topicId
        hitCount
        reactions { count reactionType }
        tags { name slug tagType }
      }
    }
    """

    COMMENTS_QUERY = """
    query questionDiscussComments($topicId: Int!, $orderBy: String = "newest_to_oldest", $pageNo: Int = 1, $numPerPage: Int = 10) {
      topicComments(
        topicId: $topicId
        orderBy: $orderBy
        pageNo: $pageNo
        numPerPage: $numPerPage
      ) {
        data {
          id
          pinned
          post {
            id
            voteCount
            voteUpCount
            content
            updationDate
            creationDate
            status
            isHidden
            anonymous
            author {
              username
              profile { userAvatar reputation realName }
              isActive
            }
          }
          numChildren
        }
        totalNum
      }
    }
    """

    @classmethod
    def get_today_str(cls) -> str:
        return date.today().isoformat()

    @classmethod
    def get_batch_id(cls, scrape_type: str) -> str:
        suffix = "bulk" if scrape_type == "bulk" else "weekly"
        return f"{cls.get_today_str()}_{suffix}"


# ============== DATA MODELS ==============
@dataclass
class InterviewDocument:
    """Unified document schema — identical to GFG scraper."""

    document_id: str
    source_platform: str
    source_url: str
    title: str
    raw_content: str
    published_at: Optional[str]
    scraped_at: str
    scrape_type: str
    scrape_batch_id: str
    source_metadata: Dict

    @property
    def content_hash(self) -> str:
        return hashlib.md5(f"{self.title}{self.raw_content}".encode()).hexdigest()

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["content_hash"] = self.content_hash
        return d

    @staticmethod
    def generate_document_id(topic_id: str, slug: str = "") -> str:
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug)[:80].strip("_").lower()
        if safe_slug:
            return f"lc_{topic_id}_{safe_slug}"
        return f"lc_{topic_id}"


@dataclass
class Manifest:
    """Scrape manifest — matches GFG manifest format."""

    scrape_date: str
    scrape_type: str
    started_at: str
    completed_at: Optional[str]
    sources: Dict
    total_files: int
    last_scraped_timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, storage: StorageBackend, path: str):
        storage.write_json(path, self.to_dict())

    @classmethod
    def load(cls, storage: StorageBackend, path: str) -> Optional["Manifest"]:
        data = storage.read_json(path)
        if data:
            return cls(**data)
        return None

    @classmethod
    def get_latest(cls, storage: StorageBackend, manifests_prefix: str) -> Optional["Manifest"]:
        files = storage.list_files(prefix=manifests_prefix, suffix=".json")
        lc_manifests = [f for f in files if "leetcode_scrape_" in f]
        if lc_manifests:
            return cls.load(storage, lc_manifests[0])
        return None


# ============== LEETCODE SCRAPER CLASS ==============
class LeetCodeScraper:
    """
    LeetCode Interview Experience Scraper.
    Refactored from standalone script to match GFG scraper class pattern.
    Accepts a StorageBackend for flexible output (local or GCS).
    """

    # ── Known companies for extraction ──
    SKIP_TAGS = frozenset({
        "interview", "interview-experience", "interview-question",
        "compensation", "system-design", "behavioral", "offer",
        "rejected", "oa", "online-assessment", "career", "feedback",
        "job-search-2", "hiring-freshers", "trending-2", "react",
        "frontend", "full-stack", "backend", "trending",
    })

    KNOWN_COMPANIES = [
        "Google", "Amazon", "Meta", "Facebook", "Microsoft", "Apple",
        "Netflix", "Uber", "Lyft", "Airbnb", "Stripe", "Coinbase",
        "Walmart", "Salesforce", "Oracle", "Adobe", "Nvidia", "Intel",
        "Tesla", "Twitter", "LinkedIn", "Snap", "TikTok", "ByteDance",
        "Atlassian", "Dropbox", "Spotify", "DoorDash", "Instacart",
        "IBM", "Cisco", "VMware", "PayPal", "Square", "Block",
        "Robinhood", "Palantir", "Snowflake", "Databricks", "MongoDB",
        "Intuit", "Visa", "Mastercard", "Goldman Sachs", "JPMorgan",
        "Morgan Stanley", "Deloitte", "Accenture", "TCS", "Infosys",
        "Wipro", "Swiggy", "Zomato", "Flipkart", "Razorpay",
        "Samsung", "Shopify", "Pinterest", "Reddit", "Discord",
        "Applied Intuition", "Two Sigma", "Citadel", "DE Shaw",
        "Jane Street", "HRT", "Tower Research",
    ]

    def __init__(
        self,
        scrape_type: str = None,
        config: ScraperConfig = None,
        storage: StorageBackend = None,
        fetch_comments: bool = False,
    ):
        self.config = config or ScraperConfig()
        self.scrape_type = scrape_type or self.config.SCRAPE_TYPE
        self.batch_id = self.config.get_batch_id(self.scrape_type)
        self.storage = storage
        self.fetch_comments = fetch_comments

        # Storage paths
        self.today_raw_prefix = f"{self.config.RAW_PREFIX}/{self.config.get_today_str()}"
        self.manifests_prefix = self.config.MANIFESTS_PREFIX

        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": self.config.USER_AGENT,
            "Referer": "https://leetcode.com/discuss/",
            "Origin": "https://leetcode.com",
        })

        # Stats
        self.stats = {
            "posts_listed": 0,
            "files_collected": 0,
            "errors": 0,
            "error_ids": [],
            "pages_fetched": 0,
        }

    # ─────────── HTTP / GraphQL ───────────
    def _graphql_request(self, operation: str, query: str, variables: Dict) -> Optional[Dict]:
        """Execute a GraphQL request. Handles rate limiting with retry."""
        payload = {
            "operationName": operation,
            "query": query,
            "variables": variables,
        }
        try:
            time.sleep(self.config.DELAY_SECONDS)
            resp = self.session.post(self.config.GRAPHQL_URL, json=payload, timeout=20)

            if resp.status_code == 429:
                print("    RATE LIMITED — waiting 60s...")
                time.sleep(60)
                resp = self.session.post(self.config.GRAPHQL_URL, json=payload, timeout=20)

            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                print(f"    GraphQL errors: {data['errors']}")
                return None

            return data.get("data")

        except requests.exceptions.Timeout:
            print(f"    TIMEOUT on {operation}")
            self.stats["errors"] += 1
            return None
        except requests.RequestException as e:
            print(f"    ERROR in {operation}: {e}")
            self.stats["errors"] += 1
            return None

    # ─────────── Step 1: List Posts (like _get_sitemaps + _extract_interview_urls) ───────────
    def _fetch_post_list(self) -> List[Dict]:
        """Fetch interview-tagged posts via discussPostItems. Paginated."""
        max_posts = (
            self.config.BULK_MAX_POSTS
            if self.scrape_type == "bulk"
            else self.config.INCREMENTAL_MAX_POSTS
        )
        print(f"\n[1/3] Fetching post listings (max {max_posts})...")

        all_posts = []
        skip = 0

        while len(all_posts) < max_posts:
            print(f"    Fetching posts {skip}-{skip + self.config.PAGE_SIZE}...", end="", flush=True)

            variables = {
                "orderBy": "HOT",
                "keywords": [""],
                "tagSlugs": self.config.TARGET_TAGS,
                "skip": skip,
                "first": min(self.config.PAGE_SIZE, max_posts - len(all_posts)),
            }

            data = self._graphql_request("discussPostItems", self.config.LIST_QUERY, variables)
            if not data:
                print(" FAILED")
                break

            articles = data.get("ugcArticleDiscussionArticles", {})
            edges = articles.get("edges", [])

            if not edges:
                print(" no more posts")
                break

            for edge in edges:
                node = edge.get("node", {})
                if node:
                    all_posts.append(node)

            self.stats["pages_fetched"] += 1
            has_next = articles.get("pageInfo", {}).get("hasNextPage", False)
            total_available = articles.get("totalNum", "?")

            print(f" ✅ {len(edges)} posts (total: {len(all_posts)}, available: {total_available})")

            skip += self.config.PAGE_SIZE
            if not has_next or len(all_posts) >= max_posts:
                break

        all_posts = all_posts[:max_posts]
        self.stats["posts_listed"] = len(all_posts)
        print(f"    Total posts listed: {len(all_posts)}")
        return all_posts

    # ─────────── Step 2: Filter (like _filter_sitemaps) ───────────
    def _filter_posts(self, posts: List[Dict]) -> List[Dict]:
        """For incremental scraping, filter to posts newer than last manifest."""
        if self.scrape_type != "incremental":
            return posts

        last_manifest = Manifest.get_latest(self.storage, self.manifests_prefix)
        if not last_manifest or not last_manifest.last_scraped_timestamp:
            print("    No previous manifest found, processing all posts.")
            return posts

        cutoff = last_manifest.last_scraped_timestamp
        print(f"\n[Filter] Incremental cutoff: {cutoff}")

        filtered = [
            p for p in posts
            if (p.get("createdAt") or p.get("updatedAt", "")) > cutoff
        ]
        print(f"    Filtered {len(posts)} → {len(filtered)} new posts")
        return filtered

    # ─────────── Step 3: Fetch Details & Save (like _scrape_articles) ───────────
    def _scrape_articles(self, posts: List[Dict]) -> List[InterviewDocument]:
        """Fetch full content for each post, parse, and save."""
        print(f"\n[2/3] Fetching full content for {len(posts)} posts...")

        documents = []

        for i, post in enumerate(posts):
            topic_id = str(post.get("topicId", ""))
            title = post.get("title", "Untitled")
            slug = post.get("slug", "")

            if not topic_id:
                continue

            print(f"    [{i+1}/{len(posts)}] {title[:55]}...", end="", flush=True)

            # Dedup check
            doc_id = InterviewDocument.generate_document_id(topic_id, slug)
            existing_path = f"{self.today_raw_prefix}/{doc_id}.json"
            if self.storage.file_exists(existing_path):
                print(" (already scraped)")
                continue

            # Fetch full content
            time.sleep(self.config.DETAIL_DELAY_SECONDS)
            data = self._graphql_request(
                "discussPostDetail", self.config.DETAIL_QUERY, {"topicId": topic_id}
            )

            if not data:
                self.stats["error_ids"].append(topic_id)
                print(" FAILED")
                continue

            article = data.get("ugcArticleDiscussionArticle")
            if not article or not article.get("content"):
                self.stats["errors"] += 1
                self.stats["error_ids"].append(topic_id)
                print(" no content")
                continue

            # Optionally fetch comments
            comments = []
            if self.fetch_comments and post.get("topic", {}).get("topLevelCommentCount", 0) > 0:
                time.sleep(self.config.DELAY_SECONDS)
                comments = self._fetch_post_comments(int(topic_id))

            doc = self._parse_article(article, post, comments)
            if doc:
                self._save_document(doc)
                documents.append(doc)
                self.stats["files_collected"] += 1
                print(" ✅")
            else:
                self.stats["errors"] += 1
                self.stats["error_ids"].append(topic_id)
                print(" parse failed")

            # Checkpoint save every 20 posts
            if (i + 1) % 20 == 0:
                print(f"    --- checkpoint: {self.stats['files_collected']} files saved ---")

        return documents

    # ─────────── Comments ───────────
    def _fetch_post_comments(self, topic_id: int, max_pages: int = 5, per_page: int = 20) -> List[Dict]:
        """Fetch comments for a single post. Mirrors standalone fetch_comments()."""
        all_comments = []
        page = 1

        while page <= max_pages:
            data = self._graphql_request(
                "questionDiscussComments",
                self.config.COMMENTS_QUERY,
                {
                    "topicId": topic_id,
                    "orderBy": "best",
                    "pageNo": page,
                    "numPerPage": per_page,
                },
            )
            if not data:
                break

            result = data.get("topicComments", {})
            comments_data = result.get("data", [])
            total_num = result.get("totalNum", 0)

            for comment in comments_data:
                post = comment.get("post", {})
                if post and not post.get("isHidden"):
                    author_data = post.get("author") or {}
                    all_comments.append({
                        "id": comment.get("id"),
                        "content": self._clean_content(post.get("content", "")),
                        "author": author_data.get("username", "anonymous"),
                        "created_at": post.get("creationDate"),
                        "vote_count": post.get("voteCount", 0),
                        "is_anonymous": post.get("anonymous", False),
                        "num_replies": comment.get("numChildren", 0),
                    })

            if len(all_comments) >= total_num or not comments_data:
                break
            page += 1

        return all_comments

    # ─────────── Parsing (like _parse_article in GFG) ───────────
    def _parse_article(self, article: Dict, list_meta: Dict, comments: List[Dict] = None) -> Optional[InterviewDocument]:
        """Parse a LeetCode article into InterviewDocument."""
        topic_id = str(article.get("topicId", ""))
        slug = article.get("slug", "")
        title = article.get("title", "Untitled")
        content = article.get("content", "")

        if not content:
            return None

        raw_content = self._clean_content(content)
        if not raw_content or len(raw_content.strip()) < 50:
            return None

        # Source URL
        source_url = f"https://leetcode.com/discuss/post/{topic_id}/{slug}/"

        # Extract metadata from list_meta (the listing response)
        tags = list_meta.get("tags", [])
        tag_names = [t.get("name", "") for t in tags if t.get("name")]
        tag_slugs = [t.get("slug", "") for t in tags if t.get("slug")]

        # Company extraction: from tags first, then title
        company = self._extract_company_from_tags(tags)
        if not company:
            company = self._extract_company_from_title(title)

        author = list_meta.get("author") or {}
        reactions = {
            r.get("reactionType", "UNKNOWN"): r.get("count", 0)
            for r in article.get("reactions", [])
        }

        return InterviewDocument(
            document_id=InterviewDocument.generate_document_id(topic_id, slug),
            source_platform="leetcode",
            source_url=source_url,
            title=title,
            raw_content=raw_content,
            published_at=article.get("createdAt"),
            scraped_at=InterviewDocument.now_iso(),
            scrape_type=self.scrape_type,
            scrape_batch_id=self.batch_id,
            source_metadata={
                "topic_id": topic_id,
                "slug": slug,
                "tags": tag_names,
                "tag_slugs": tag_slugs,
                "company": company,
                "author_username": author.get("userName", "anonymous"),
                "is_anonymous": list_meta.get("isAnonymous", True),
                "hit_count": article.get("hitCount", 0),
                "comment_count": (list_meta.get("topic") or {}).get("topLevelCommentCount", 0),
                "reactions": reactions,
                "created_at": article.get("createdAt"),
                "updated_at": article.get("updatedAt"),
                "comments": comments or [],
            },
        )

    # ─────────── Content Cleaning ───────────
    def _clean_content(self, content: str) -> str:
        """Clean HTML/markdown content."""
        if not content:
            return ""
        text = re.sub(r"<[^>]+>", "\n", content)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    # ─────────── Company Extraction ───────────
    def _extract_company_from_tags(self, tags: List[Dict]) -> Optional[str]:
        """Extract company name from post tags."""
        # First: look for COMPANY type tags
        for tag in tags:
            if tag.get("tagType") == "COMPANY":
                return tag.get("name")

        # Second: match known companies from slug
        known_lower = {c.lower(): c for c in self.KNOWN_COMPANIES}
        for tag in tags:
            slug = tag.get("slug", "").lower()
            if slug in known_lower and slug not in self.SKIP_TAGS:
                return known_lower[slug]

        return None

    def _extract_company_from_title(self, title: str) -> Optional[str]:
        """Fallback: extract company name from post title."""
        title_lower = title.lower()
        for company in self.KNOWN_COMPANIES:
            if company.lower() in title_lower:
                return company
        return None

    # ─────────── Storage (like _save_document in GFG) ───────────
    def _save_document(self, doc: InterviewDocument):
        """Save document via storage backend."""
        path = f"{self.today_raw_prefix}/{doc.document_id}.json"
        self.storage.write_json(path, doc.to_dict())

    # ─────────── Manifest (like _create_manifest in GFG) ───────────
    def _create_manifest(self, posts: List[Dict], started_at: str) -> Manifest:
        """Create and save manifest for this scrape run."""
        latest_ts = None
        for p in posts:
            ts = p.get("createdAt") or p.get("updatedAt")
            if ts and (latest_ts is None or ts > latest_ts):
                latest_ts = ts

        manifest = Manifest(
            scrape_date=self.config.get_today_str(),
            scrape_type=self.scrape_type,
            started_at=started_at,
            completed_at=InterviewDocument.now_iso(),
            sources={
                "leetcode": {
                    "files_collected": self.stats["files_collected"],
                    "posts_listed": self.stats["posts_listed"],
                    "pages_fetched": self.stats["pages_fetched"],
                    "errors": self.stats["errors"],
                    "error_ids": self.stats["error_ids"][:10],
                }
            },
            total_files=self.stats["files_collected"],
            last_scraped_timestamp=latest_ts,
        )

        manifest_path = f"{self.manifests_prefix}/leetcode_scrape_{self.config.get_today_str()}.json"
        manifest.save(self.storage, manifest_path)
        print(f"\nManifest saved to {manifest_path}")
        return manifest

    # ─────────── Summary (bonus: company stats from standalone script) ───────────
    def _print_summary(self, documents: List[InterviewDocument]):
        """Print scrape summary with company stats."""
        print("\n" + "=" * 60)
        print("SCRAPE COMPLETE")
        print("=" * 60)
        print(f"  Posts listed:    {self.stats['posts_listed']}")
        print(f"  Files collected: {self.stats['files_collected']}")
        print(f"  Errors:          {self.stats['errors']}")

        if documents:
            companies = [
                d.source_metadata.get("company")
                for d in documents if d.source_metadata.get("company")
            ]
            if companies:
                print("\n  Top companies mentioned:")
                for company, cnt in Counter(companies).most_common(10):
                    print(f"    - {company}: {cnt}")

            dates = [d.published_at for d in documents if d.published_at]
            if dates:
                print(f"\n  Date range: {min(dates)[:10]} to {max(dates)[:10]}")

    # ─────────── Main Entry Point (like run() in GFG) ───────────
    def run(self):
        """Execute the scraping pipeline."""
        print("=" * 60)
        print(f"LeetCode Interview Scraper - {self.scrape_type.upper()} mode")
        print(f"Storage: {self.storage.__class__.__name__}")
        print(f"Comments: {'ON' if self.fetch_comments else 'OFF'}")
        print("=" * 60)
        print(f"Batch ID: {self.batch_id}")

        started_at = InterviewDocument.now_iso()

        # Step 1: List posts
        posts = self._fetch_post_list()
        if not posts:
            print("ERROR: No posts found. Exiting.")
            return

        # Step 2: Filter (incremental only)
        posts = self._filter_posts(posts)
        if not posts:
            print("No new posts to process. Exiting.")
            return

        # Step 3: Fetch details & save
        documents = self._scrape_articles(posts)

        # Create manifest
        self._create_manifest(posts, started_at)

        # Summary
        self._print_summary(documents)


# ============== ENTRY POINT ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LeetCode Interview Scraper")
    parser.add_argument("--type", choices=["bulk", "incremental"], default="bulk")
    parser.add_argument("--count", type=int, default=100, help="Max posts to scrape")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    parser.add_argument("--with-comments", action="store_true", help="Also fetch comments")
    parser.add_argument("--gcs-bucket", type=str, default=None, help="GCS bucket name")
    parser.add_argument("--gcs-creds", type=str, default=None, help="GCS credentials JSON path")
    args = parser.parse_args()

    # ── Storage backend selection ──
    if args.gcs_bucket:
        from src.storage.gcs_backend import GCSBackend
        storage = GCSBackend(bucket_name=args.gcs_bucket, credentials_path=args.gcs_creds)
    else:
        # Quick local storage for development
        import os

        class LocalStorageBackend(StorageBackend):
            def __init__(self, base_dir="data"):
                self.base = base_dir

            def write_json(self, path, data):
                full = os.path.join(self.base, path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            def read_json(self, path):
                full = os.path.join(self.base, path)
                if not os.path.exists(full):
                    return None
                with open(full, "r", encoding="utf-8") as f:
                    return json.load(f)

            def file_exists(self, path):
                return os.path.exists(os.path.join(self.base, path))

            def list_files(self, prefix="", suffix=""):
                results = []
                base_path = os.path.join(self.base, prefix)
                if os.path.exists(base_path):
                    for root, _, files in os.walk(base_path):
                        for fname in files:
                            if not suffix or fname.endswith(suffix):
                                rel = os.path.relpath(os.path.join(root, fname), self.base)
                                results.append(rel)
                return sorted(results, reverse=True)

        storage = LocalStorageBackend(base_dir="data")

    # Override config with CLI args
    config = ScraperConfig()
    config.DELAY_SECONDS = args.delay
    config.DETAIL_DELAY_SECONDS = args.delay
    if args.type == "bulk":
        config.BULK_MAX_POSTS = args.count
    else:
        config.INCREMENTAL_MAX_POSTS = args.count

    scraper = LeetCodeScraper(
        scrape_type=args.type,
        config=config,
        storage=storage,
        fetch_comments=args.with_comments,
    )
    scraper.run()