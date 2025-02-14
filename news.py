import os
import requests
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Get API credentials from environment variables
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")

# API Base URL
ASKNEWS_API_URL = "https://api.asknews.app/v1/news/search"

def fetch_enriched_news(
    query: str,
    num_articles: int = 5,
    historical: bool = False,
    strategy: str = "latest news",
    sentiment: str = None,
    categories: list[str] = None,
    countries: list[str] = None,
    languages: list[str] = None,
):
    """
    Fetches enriched real-time news from the AskNews API.

    Parameters:
        query (str): Search query string (keywords, phrases, or natural language)
        num_articles (int): Number of articles to return (default: 5)
        historical (bool): Search historical news archive (default: False, last 48h)
        strategy (str): Search strategy ('latest news', 'news knowledge', 'default')
        sentiment (str): Filter articles by sentiment (e.g., 'positive', 'negative')
        categories (list[str]): Filter by news categories (e.g., ['Science', 'Politics'])
        countries (list[str]): Filter by country codes (e.g., ['US', 'GB'])
        languages (list[str]): Filter by language codes (e.g., ['en', 'fr'])

    Returns:
        list: A list of articles with key details.
    """

    if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
        raise ValueError("AskNews API credentials not found in .env file.")

    headers = {
        "Authorization": f"Bearer {ASKNEWS_SECRET}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Construct request parameters
    payload = {
        "query": query,
        "n_articles": num_articles,
        "return_type": "dicts",  # Get structured metadata-rich response
        "historical": historical,
        "strategy": strategy,  # "latest news" (past 24h) or "news knowledge" (past 60 days)
    }

    if sentiment:
        payload["sentiment"] = sentiment
    if categories:
        payload["categories"] = categories
    if countries:
        payload["countries"] = countries
    if languages:
        payload["languages"] = languages

    response = requests.get(ASKNEWS_API_URL, headers=headers, params=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Error fetching articles: {response.status_code}, {response.text}")

    articles = response.json()

    # Extract relevant fields from API response
    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            "Title": article["title"],
            "Summary": article["summary"],
            "Source": article["source_id"],
            "Country": article["country"],
            "Language": article["language"],
            "Publication Date": article["pub_date"],
            "Article URL": article["article_url"],
            "Sentiment": article.get("sentiment", "N/A"),
            "Keywords": article.get("keywords", []),
            "Classification": article.get("classification", "N/A"),
        })

    return formatted_articles

# Example Usage
if __name__ == "__main__":
    search_query = "Artificial Intelligence"
    articles = fetch_enriched_news(
        query=search_query, num_articles=5, categories=["Technology", "Science"]
    )

    print("\n🔹 Enriched News Articles:")
    for idx, article in enumerate(articles, start=1):
        print(f"\n🔹 Article {idx}:")
        for key, value in article.items():
            print(f"   {key}: {value}")
