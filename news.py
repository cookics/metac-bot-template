"""
News/research functionality using EXA.
"""
import asyncio
import forecasting_tools
from config import GET_NEWS, OPENAI_API_KEY


def call_exa_smart_searcher(question: str) -> str:
    """
    Search for relevant news using EXA.
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
    Returns research summary or 'No research done' if GET_NEWS is False.
    """
    if GET_NEWS:
        research = call_exa_smart_searcher(question)
    else:
        research = "No research done"

    print(f"########################\nResearch Found:\n{research}\n########################")
    return research
