"""
News/research functionality using EXA.
"""
import asyncio
from exa_py import Exa
try:
    import forecasting_tools
    HAS_FORECASTING_TOOLS = True
except ImportError:
    HAS_FORECASTING_TOOLS = False

from config import GET_NEWS, OPENAI_API_KEY, EXA_API_KEY, USE_SMART_SEARCHER


def exa_search_raw(
    query: str, 
    num_results: int = 10,
    end_published_date: str = None,
    start_published_date: str = None,
    return_cost: bool = False
) -> list[dict] | tuple[list[dict], dict]:
    """
    Perform a search using Exa API and return raw results as list of dicts.
    This is used by the research agent to filter relevant results.
    
    Args:
        query: Search query
        num_results: Number of results to return
        end_published_date: Only return results published before this date (ISO format)
        start_published_date: Only return results published after this date (ISO format)
        return_cost: If True, return (results, cost_info) tuple
    """
    if not EXA_API_KEY:
        return ([], {"total": 0.0}) if return_cost else []

    exa = Exa(api_key=EXA_API_KEY)
    
    # Build search parameters
    search_params = {
        "query": query,
        "num_results": num_results,
        "text": True,
        "type": "auto",
    }
    
    # Add date filters for backtesting (no future peeking)
    if end_published_date:
        search_params["end_published_date"] = end_published_date
    if start_published_date:
        search_params["start_published_date"] = start_published_date
    
    result = exa.search_and_contents(**search_params)

    raw_results = []
    for i, res in enumerate(result.results):
        raw_results.append({
            "index": i,
            "title": res.title,
            "url": res.url,
            "score": res.score,
            "published_date": res.published_date,
            "text": res.text[:1000] if res.text else "",
            "highlights": res.highlights if res.highlights else [],
        })
    
    if return_cost:
        # Extract cost from response if available
        cost_info = {"total": 0.0}
        if hasattr(result, 'cost_dollars') and result.cost_dollars:
            # CostDollars is an object, access via attributes
            cost_obj = result.cost_dollars
            try:
                cost_info = {
                    "total": getattr(cost_obj, 'total', 0.0) or 0.0,
                }
            except Exception:
                cost_info = {"total": 0.0}
        return raw_results, cost_info
    
    return raw_results


def exa_search_and_contents(query: str, num_results: int = 10) -> str:
    """
    Perform a search and get contents using the pure Exa API.
    """
    if not EXA_API_KEY:
        return "EXA_API_KEY not found in configuration."

    exa = Exa(api_key=EXA_API_KEY)
    
    # Using deep search as requested
    result = exa.search_and_contents(
        query,
        context=True,
        num_results=num_results,
        text=True,
        type="deep",
        user_location="US"
    )

    combined_results = ""
    for i, res in enumerate(result.results):
        combined_results += (
            f'[Result {i+1}]:\n'
            f'Title: {res.title}\n'
            f'URL: {res.url}\n'
            f'Score: {res.score}\n'
            f'Published Date: {res.published_date}\n'
            f'Text Snippet: {res.text[:500]}...\n'
            f'Highlights: {" ".join(res.highlights) if res.highlights else "No highlights"}\n\n'
        )
    return combined_results


def exa_crawl_urls(urls: list[str]) -> list[dict]:
    """
    Crawl URLs and return raw content as list of dicts.
    Used by the research agent to get full page content.
    """
    if not EXA_API_KEY or not urls:
        return []

    exa = Exa(api_key=EXA_API_KEY)
    
    try:
        result = exa.get_contents(urls, text=True)
        crawled = []
        for res in result.results:
            crawled.append({
                "url": res.url,
                "title": res.title,
                "text": res.text[:3000] if res.text else "",  # Limit content size
            })
        return crawled
    except Exception as e:
        print(f"[Exa Crawl] Error crawling URLs: {e}")
        return []


def exa_get_contents(urls: list[str]) -> str:
    """
    Retrieve webpage contents for a list of URLs using the pure Exa API.
    """
    if not EXA_API_KEY:
        return "EXA_API_KEY not found in configuration."

    exa = Exa(api_key=EXA_API_KEY)
    result = exa.get_contents(urls, text=True)

    combined_contents = ""
    for i, res in enumerate(result.results):
        combined_contents += (
            f'[Crawl Result {i+1}]:\n'
            f'Title: {res.title}\n'
            f'URL: {res.url}\n'
            f'Content: {res.text[:1000]}...\n\n'
        )
    return combined_contents


def call_exa_smart_searcher(question: str) -> str:
    """
    Search for relevant news using EXA via forecasting-tools.
    Uses SmartSearcher if OPENAI_API_KEY is available, otherwise basic ExaSearcher.
    """
    if not HAS_FORECASTING_TOOLS:
        return "forecasting-tools package not installed. Cannot use SmartSearcher."

    if OPENAI_API_KEY is None:
        searcher = forecasting_tools.ExaSearcher(
            include_highlights=True,
            num_results=10,
        )
        highlights = asyncio.run(
            searcher.invoke_for_highlights_in_relevance_order(question)
        )
        prioritized_highlights = highlights[:10]
        combined_highlights = ""
        for i, highlight in enumerate(prioritized_highlights):
            combined_highlights += (
                f'[Highlight {i+1}]:\n'
                f'Title: {highlight.source.title}\n'
                f'URL: {highlight.source.url}\n'
                f'Text: "{highlight.highlight_text}"\n\n'
            )
        response = combined_highlights
    else:
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

    return response


def run_research(question: str) -> str:
    """
    Run research for a question using EXA.
    Uses pure Exa search as the default (controlled by USE_SMART_SEARCHER).
    """
    if GET_NEWS:
        if USE_SMART_SEARCHER:
            print(f"Running SmartSearcher for: {question}")
            research = call_exa_smart_searcher(question)
        else:
            print(f"Running pure Exa search for: {question}")
            research = exa_search_and_contents(question)
    else:
        research = "No research done"

    print(f"########################\nResearch Found:\n{research}\n########################")
    return research


if __name__ == "__main__":
    test_query = "blog post about artificial intelligence"
    print(f"--- Testing Exa Search for: '{test_query}' ---")
    search_results = exa_search_and_contents(test_query, num_results=3)
    print(search_results)

    test_urls = ["https://exa.ai"]
    print(f"--- Testing Exa Get Contents (Crawl) for: {test_urls} ---")
    crawl_results = exa_get_contents(test_urls)
    print(crawl_results)
