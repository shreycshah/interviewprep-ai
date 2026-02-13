from datetime import date


class LeetCodeScraperConfigs:
    """Configuration for LeetCode Scraper — mirrors GFGScraperConfigs."""

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

    RAW_PREFIX = "raw"

    # ── Tags to skip during company extraction ──
    SKIP_TAGS = frozenset({
        "interview", "interview-experience", "interview-question",
        "compensation", "system-design", "behavioral", "offer",
        "rejected", "oa", "online-assessment", "career", "feedback",
        "job-search-2", "hiring-freshers", "trending-2", "react",
        "frontend", "full-stack", "backend", "trending",
    })

    # ── Known companies for extraction ──
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

    @classmethod
    def get_raw_prefix(cls, scrape_type: str) -> str:
        """
        Returns the raw prefix based on scrape type:
        - bulk: raw/bulk/leetcode
        - incremental: raw/incremental/{date}/leetcode
        """
        if scrape_type == "bulk":
            return f"{cls.RAW_PREFIX}/bulk/leetcode"
        else:
            return f"{cls.RAW_PREFIX}/incremental/{cls.get_today_str()}/leetcode"