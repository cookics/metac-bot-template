"""
News/research functionality using EXA.
"""
import asyncio
from exa_py import Exa
import forecasting_tools
from config import GET_NEWS, OPENAI_API_KEY, EXA_API_KEY, USE_SMART_SEARCHER


def exa_search_raw(query: str, num_results: int = 10) -> list[dict]:
    """
    Perform a search using Exa API and return raw results as list of dicts.
    This is used by the research agent to filter relevant results.
    """
    if not EXA_API_KEY:
        return []

    exa = Exa(api_key=EXA_API_KEY)
    
    result = exa.search_and_contents(
        query,
        context=True,
        num_results=num_results,
        text=True,
        type="deep",
        user_location="US"
    )

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
