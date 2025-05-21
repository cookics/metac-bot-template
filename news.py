import os
import requests
import dotenv
import asyncio
import forecasting_tools # Assuming forecasting_tools is a custom library available

# Load environment variables from .env file
dotenv.load_dotenv()

from config import OPENAI_API_KEY # Import OPENAI_API_KEY from config

# Get API credentials from environment variables
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
# OPENAI_API_KEY is now imported from config
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
# OPENAI_API_KEY is imported from config at the top of the file

# API Base URL
ASKNEWS_API_URL = "https://api.asknews.app/v1/news/search"

def fetch_asknews_articles(
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


def fetch_exa_articles(question: str) -> str:
    if not EXA_API_KEY:
        raise ValueError("Exa API key not found in .env file.")

    if OPENAI_API_KEY: # Use SmartSearcher if OpenAI key is available
        searcher = forecasting_tools.SmartSearcher(
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give "
            "you a question they intend to forecast on. To be a great assistant, you generate "
            "a concise but detailed rundown of the most relevant news, including if the question "
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )
        response = asyncio.run(searcher.invoke(prompt))
    else:
        searcher = forecasting_tools.ExaSearcher(
            include_highlights=True,
            num_results=10,
        )
        highlights = asyncio.run(searcher.invoke_for_highlights_in_relevance_order(question))
        prioritized_highlights = highlights[:10]
        combined_highlights = ""
        for i, highlight in enumerate(prioritized_highlights):
            combined_highlights += f'[Highlight {i+1}]:\nTitle: {highlight.source.title}\nURL: {highlight.source.url}\nText: "{highlight.highlight_text}"\n\n'
        response = combined_highlights
    return response

def fetch_perplexity_articles(question: str) -> str: # Note: OPENAI_API_KEY not used here
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key not found in .env file.")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3.1-sonar-huge-128k-online",
        "messages": [
            {
                "role": "system",
                "content": """
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.
                """,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    }
    response = requests.post(url=url, json=payload, headers=headers)
    if not response.ok:
        raise RuntimeError(f"Error fetching Perplexity articles: {response.status_code}, {response.text}")
    content = response.json()["choices"][0]["message"]["content"]
    return content


def get_research_report(question: str, provider: str = "asknews", enable_research: bool = True) -> str:
    """
    Fetches research information from the specified provider and formats it into a string report.
    If enable_research is False, returns a message indicating research is disabled.
    """
    if not enable_research:
        return "Research disabled by configuration."

    report = f"Research Report for: {question}\nProvider: {provider.capitalize()}\n\n"

    # Ensure API keys are loaded if not already. This is a safeguard.
    # Actual loading should occur at module import or app start.
    # dotenv.load_dotenv() # Already called at the top of the module

    # Re-check API keys from environment in case they weren't loaded at module level
    # or to ensure they are fresh if they can change during runtime (unlikely for this app)
    # This part might be redundant if module-level loading is guaranteed and sufficient.
    current_asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
    current_asknews_secret = os.getenv("ASKNEWS_SECRET")
    current_exa_api_key = os.getenv("EXA_API_KEY")
    # OPENAI_API_KEY is imported from config
    current_perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")


    if provider == "asknews":
        # Use current_asknews_client_id and current_asknews_secret
        if not current_asknews_client_id or not current_asknews_secret:
            return report + "AskNews API credentials not configured."
        try:
            # Using the renamed function fetch_asknews_articles
            articles = fetch_asknews_articles(query=question, num_articles=5) # Defaulting to 5 articles for now
            if articles:
                formatted_articles_list = []
                for article in articles:
                    article_details = []
                    for key, value in article.items():
                        article_details.append(f"   {key}: {value}")
                    formatted_articles_list.append("\n🔹 Article:\n" + "\n".join(article_details))
                report += "\n".join(formatted_articles_list)
            else:
                report += "No articles found by AskNews."
        except Exception as e:
            report += f"Error fetching AskNews articles: {e}"

    elif provider == "exa":
        # Use current_exa_api_key and OPENAI_API_KEY (imported)
        if not current_exa_api_key:
            return report + "Exa API key not configured."
        try:
            # fetch_exa_articles uses EXA_API_KEY and OPENAI_API_KEY directly from module/config level
            exa_results = fetch_exa_articles(question)
            report += exa_results
        except Exception as e:
            report += f"Error fetching Exa articles: {e}"

    elif provider == "perplexity":
        # Use current_perplexity_api_key
        if not current_perplexity_api_key:
            return report + "Perplexity API key not configured."
        try:
            perplexity_results = fetch_perplexity_articles(question)
            report += perplexity_results
        except Exception as e:
            report += f"Error fetching Perplexity articles: {e}"
    else:
        return report + f"Unknown provider: {provider}. Please choose from 'asknews', 'exa', or 'perplexity'."

    return report


# Example Usage
if __name__ == "__main__":
    search_query = "Latest breakthroughs in quantum computing"

    # Test AskNews
    print("--- Testing AskNews ---")
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET: # Check module level loaded keys for example
        asknews_report = get_research_report(search_query, provider="asknews")
        print(asknews_report)
    else:
        print("AskNews API credentials not set. Skipping AskNews test.")

    # Test Exa
    print("\n--- Testing Exa ---")
    if EXA_API_KEY: # Check module level loaded keys for example
        exa_report = get_research_report(search_query, provider="exa")
        print(exa_report)
    else:
        print("Exa API key not set. Skipping Exa test.")

    # Test Perplexity
    print("\n--- Testing Perplexity ---")
    if PERPLEXITY_API_KEY: # Check module level loaded keys for example
        perplexity_report = get_research_report(search_query, provider="perplexity")
        print(perplexity_report)
    else:
        print("Perplexity API key not set. Skipping Perplexity test.")

    # Test disabled research
    print("\n--- Testing Disabled Research ---")
    disabled_report = get_research_report(search_query, provider="asknews", enable_research=False)
    print(disabled_report)
        if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
            return report + "AskNews API credentials not configured."
        try:
            # Using the renamed function fetch_asknews_articles
            articles = fetch_asknews_articles(query=question, num_articles=5) # Defaulting to 5 articles for now
            if articles:
                formatted_articles_list = []
                for article in articles:
                    article_details = []
                    for key, value in article.items():
                        article_details.append(f"   {key}: {value}")
                    formatted_articles_list.append("\n🔹 Article:\n" + "\n".join(article_details))
                report += "\n".join(formatted_articles_list)
            else:
                report += "No articles found by AskNews."
        except Exception as e:
            report += f"Error fetching AskNews articles: {e}"

    elif provider == "exa":
        if not EXA_API_KEY:
            return report + "Exa API key not configured."
        try:
            exa_results = fetch_exa_articles(question)
            report += exa_results
        except Exception as e:
            report += f"Error fetching Exa articles: {e}"

    elif provider == "perplexity":
        if not PERPLEXITY_API_KEY:
            return report + "Perplexity API key not configured."
        try:
            perplexity_results = fetch_perplexity_articles(question)
            report += perplexity_results
        except Exception as e:
            report += f"Error fetching Perplexity articles: {e}"
    else:
        return report + f"Unknown provider: {provider}. Please choose from 'asknews', 'exa', or 'perplexity'."

    return report


# Example Usage
if __name__ == "__main__":
    search_query = "Latest breakthroughs in quantum computing"

    # Test AskNews
    print("--- Testing AskNews ---")
    if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        asknews_report = get_research_report(search_query, provider="asknews")
        print(asknews_report)
    else:
        print("AskNews API credentials not set. Skipping AskNews test.")

    # Test Exa
    print("\n--- Testing Exa ---")
    if EXA_API_KEY:
        exa_report = get_research_report(search_query, provider="exa")
        print(exa_report)
    else:
        print("Exa API key not set. Skipping Exa test.")

    # Test Perplexity
    print("\n--- Testing Perplexity ---")
    if PERPLEXITY_API_KEY:
        perplexity_report = get_research_report(search_query, provider="perplexity")
        print(perplexity_report)
    else:
        print("Perplexity API key not set. Skipping Perplexity test.")
