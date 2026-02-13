"""
LeetCode Interview Experience Scraper (GCS-compatible)
=====================================================
Scrapes interview experiences from LeetCode Discuss GraphQL API.

Usage:
    from src.storage.gcs_backend import GCSBackend

    storage = GCSBackend(
        bucket_name="interviewprep-ai-data",
        project_id="professorbot-dovbsg",
        secret_name="gcs-service-account-key",
    )
    scraper = LeetCodeScraper(scrape_type="bulk", storage=storage)
    scraper.run()
"""

import requests
import json
import time
import re
from datetime import datetime
from typing import Optional, List, Dict
from collections import Counter

from src.storage.storage_backend import StorageBackend
from src.scrapers.configs.leetcode import LeetCodeScraperConfigs
from src.data_models.scraped_document import ScrapedInterviewDocument


# ============== LEETCODE SCRAPER CLASS ==============
class LeetCodeScraper:
    """
    LeetCode Interview Experience Scraper.
    Uses GraphQL API to fetch posts tagged "interview" from /discuss/.
    Accepts a StorageBackend for flexible output (local or GCS).
    """

    def __init__(
        self,
        scrape_type: str = None,
        config: LeetCodeScraperConfigs = None,
        storage: StorageBackend = None,
        fetch_comments: bool = False,
    ):
        self.config = config or LeetCodeScraperConfigs()
        self.scrape_type = scrape_type or self.config.SCRAPE_TYPE
        self.batch_id = self.config.get_batch_id(self.scrape_type)
        self.fetch_comments = fetch_comments

        if storage is None:
            raise ValueError(
                "A valid StorageBackend instance must be provided to LeetCodeScraper. "
                "Got storage=None."
            )
        self.storage = storage

        # Storage paths
        self.today_raw_prefix = self.config.get_raw_prefix(self.scrape_type)

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

    # ─────────── Step 1: List Posts ───────────
    def _fetch_post_list(self) -> List[Dict]:
        """Fetch interview-tagged posts via discussPostItems. Paginated."""
        max_posts = (
            self.config.BULK_MAX_POSTS
            if self.scrape_type == "bulk"
            else self.config.INCREMENTAL_MAX_POSTS
        )
        print(f"\n[1/2] Fetching post listings (max {max_posts})...")

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

    # ─────────── Step 2: Fetch Details & Save ───────────
    def _scrape_articles(self, posts: List[Dict]) -> List[ScrapedInterviewDocument]:
        """Fetch full content for each post, parse, and save."""
        print(f"\n[2/2] Fetching full content for {len(posts)} posts...")

        documents = []

        for i, post in enumerate(posts):
            topic_id = str(post.get("topicId", ""))
            title = post.get("title", "Untitled")
            slug = post.get("slug", "")

            if not topic_id:
                continue

            print(f"    [{i+1}/{len(posts)}] {title[:55]}...", end="", flush=True)

            # Build source URL for dedup check
            source_url = f"https://leetcode.com/discuss/post/{topic_id}/{slug}/"
            doc_id = ScrapedInterviewDocument.generate_document_id("leetcode", source_url)
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

            # Checkpoint every 20 posts
            if (i + 1) % 20 == 0:
                print(f"    --- checkpoint: {self.stats['files_collected']} files saved ---")

        return documents

    # ─────────── Comments ───────────
    def _fetch_post_comments(self, topic_id: int, max_pages: int = 5, per_page: int = 20) -> List[Dict]:
        """Fetch comments for a single post."""
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
                    all_comments.append({
                        "id": comment.get("id"),
                        "content": self._clean_content(post.get("content", "")),
                        "created_at": post.get("creationDate"),
                        "vote_count": post.get("voteCount", 0),
                        "is_anonymous": post.get("anonymous", False),
                        "num_replies": comment.get("numChildren", 0),
                    })

            if len(all_comments) >= total_num or not comments_data:
                break
            page += 1

        return all_comments

    # ─────────── Parsing ───────────
    def _parse_article(self, article: Dict, list_meta: Dict, comments: List[Dict] = None) -> Optional[ScrapedInterviewDocument]:
        """Parse a LeetCode article into ScrapedInterviewDocument."""
        topic_id = str(article.get("topicId", ""))
        slug = article.get("slug", "")
        title = article.get("title", "Untitled")
        content = article.get("content", "")

        if not content:
            return None

        raw_content = self._clean_content(content)
        if not raw_content or len(raw_content.strip()) < 50:
            return None

        source_url = f"https://leetcode.com/discuss/post/{topic_id}/{slug}/"

        # Extract metadata
        tags = list_meta.get("tags", [])
        tag_names = [t.get("name", "") for t in tags if t.get("name")]
        tag_slugs = [t.get("slug", "") for t in tags if t.get("slug")]

        company = self._extract_company_from_tags(tags)
        if not company:
            company = self._extract_company_from_title(title)

        reactions = {
            r.get("reactionType", "UNKNOWN"): r.get("count", 0)
            for r in article.get("reactions", [])
        }

        return ScrapedInterviewDocument(
            document_id=ScrapedInterviewDocument.generate_document_id("leetcode", source_url),
            source_platform="leetcode",
            source_url=source_url,
            title=title,
            raw_content=raw_content,
            published_at=article.get("createdAt"),
            scraped_at=ScrapedInterviewDocument.now_iso(),
            scrape_type=self.scrape_type,
            scrape_batch_id=self.batch_id,
            source_metadata={
                "topic_id": topic_id,
                "slug": slug,
                "tags": tag_names,
                "tag_slugs": tag_slugs,
                "company": company,
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
        for tag in tags:
            if tag.get("tagType") == "COMPANY":
                return tag.get("name")

        known_lower = {c.lower(): c for c in self.config.KNOWN_COMPANIES}
        for tag in tags:
            slug = tag.get("slug", "").lower()
            if slug in known_lower and slug not in self.config.SKIP_TAGS:
                return known_lower[slug]

        return None

    def _extract_company_from_title(self, title: str) -> Optional[str]:
        """Fallback: extract company name from post title."""
        title_lower = title.lower()
        for company in self.config.KNOWN_COMPANIES:
            if company.lower() in title_lower:
                return company
        return None

    # ─────────── Storage ───────────
    def _save_document(self, doc: ScrapedInterviewDocument):
        """Save document via storage backend."""
        path = f"{self.today_raw_prefix}/{doc.document_id}.json"
        self.storage.write_json(path, doc.to_dict())

    # ─────────── Summary ───────────
    def _print_summary(self, documents: List[ScrapedInterviewDocument], start_time: datetime):
        """Print scrape summary with company stats and timing."""
        end_time = datetime.now()
        delta = end_time - start_time
        total_seconds = int(delta.total_seconds())
        minutes, seconds = divmod(total_seconds, 60)

        print("\n" + "=" * 60)
        print("SCRAPE COMPLETE")
        print("=" * 60)
        print(f"  Posts listed:    {self.stats['posts_listed']}")
        print(f"  Files collected: {self.stats['files_collected']}")
        print(f"  Errors:          {self.stats['errors']}")
        print(f"  Total time taken for LeetCode Scraping: {minutes} min {seconds} sec")

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

    # ─────────── Main Entry Point ───────────
    def run(self):
        """Execute the scraping pipeline."""
        print("=" * 60)
        print(f"LeetCode Interview Scraper - {self.scrape_type.upper()} mode")
        print(f"Storage: {self.storage.__class__.__name__}")
        print(f"Comments: {'ON' if self.fetch_comments else 'OFF'}")
        print("=" * 60)
        print(f"Batch ID: {self.batch_id}")

        start_time = datetime.now()

        # Step 1: List posts
        posts = self._fetch_post_list()
        if not posts:
            print("ERROR: No posts found. Exiting.")
            return

        # Step 2: Fetch details & save
        documents = self._scrape_articles(posts)

        # Summary
        self._print_summary(documents, start_time)


# ============== ENTRY POINT ==============
# if __name__ == "__main__":
#     from src.storage.gcs_backend import GCSBackend
#
#     storage = GCSBackend(
#         bucket_name="interviewprep-ai-data",
#         project_id="professorbot-dovbsg",
#         secret_name="gcs-service-account-key",
#     )
#     scraper = LeetCodeScraper(scrape_type="bulk", storage=storage)
#     scraper.run()