"""
Research Agent - Filters and summarizes search results.

This agent uses a cheaper LLM to:
1. Review raw search results
2. Select only relevant results
3. Generate a brief summary for the forecasting agent
"""
import asyncio
import json
import re

from llm import call_llm
from news import exa_search_raw
from config import RESEARCH_MODEL, RESEARCH_TEMP, GET_NEWS
from prompts import RESEARCH_AGENT_PROMPT


async def run_research_agent(question: str) -> tuple[list[dict], str]:
    """
    Run the research agent to search, filter, and summarize.
    
    Returns:
        tuple: (relevant_results, summary)
            - relevant_results: List of filtered search result dicts
            - summary: Brief summary of findings for the forecaster
    """
    if not GET_NEWS:
        return [], "No research performed (GET_NEWS is disabled)."

    print(f"[Research Agent] Searching for: {question}")
    
    # Get raw search results
    raw_results = exa_search_raw(question)
    
    if not raw_results:
        return [], "No search results found."
    
    print(f"[Research Agent] Found {len(raw_results)} results, filtering...")
    
    # Format results for the LLM
    results_json = json.dumps(raw_results, indent=2, default=str)
    
    # Ask the research agent to filter and summarize
    prompt = RESEARCH_AGENT_PROMPT.format(
        question=question,
        results_json=results_json
    )
    
    response = await call_llm(
        prompt, 
        model=RESEARCH_MODEL, 
        temperature=RESEARCH_TEMP
    )
    
    # Parse the response to extract selected indices and summary
    relevant_results, summary = parse_research_agent_response(response, raw_results)
    
    print(f"[Research Agent] Selected {len(relevant_results)} relevant results")
    print(f"[Research Agent] Summary: {summary[:200]}...")
    
    return relevant_results, summary


def parse_research_agent_response(response: str, raw_results: list[dict]) -> tuple[list[dict], str]:
    """
    Parse the research agent's response to extract selected indices and summary.
    
    Expected format:
    RELEVANT_INDICES: [0, 2, 5]
    SUMMARY: Brief summary of findings...
    """
    relevant_results = []
    summary = ""
    
    # Extract indices
    indices_match = re.search(r'RELEVANT_INDICES:\s*\[([\d,\s]*)\]', response)
    if indices_match:
        indices_str = indices_match.group(1)
        try:
            indices = [int(i.strip()) for i in indices_str.split(',') if i.strip()]
            relevant_results = [raw_results[i] for i in indices if 0 <= i < len(raw_results)]
        except (ValueError, IndexError):
            # If parsing fails, return all results
            relevant_results = raw_results
    else:
        # Fallback: return all results
        relevant_results = raw_results
    
    # Extract summary
    summary_match = re.search(r'SUMMARY:\s*(.+)', response, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
    else:
        summary = response  # Use entire response as summary if parsing fails
    
    return relevant_results, summary


def format_results_for_forecaster(relevant_results: list[dict], summary: str) -> str:
    """
    Format the filtered results and summary for the forecasting agent.
    """
    output = f"Research Summary:\n{summary}\n\n"
    output += "Relevant Sources:\n"
    
    for i, result in enumerate(relevant_results):
        output += (
            f"[{i+1}] {result['title']}\n"
            f"    URL: {result['url']}\n"
            f"    Published: {result.get('published_date', 'Unknown')}\n"
            f"    Content: {result['text'][:500]}...\n\n"
        )
    
    return output


# Backward-compatible wrapper for existing code
def run_research(question: str) -> str:
    """
    Wrapper that runs the research agent and returns formatted results.
    This maintains backward compatibility with existing forecasting code.
    """
    relevant_results, summary = asyncio.run(run_research_agent(question))
    return format_results_for_forecaster(relevant_results, summary)
