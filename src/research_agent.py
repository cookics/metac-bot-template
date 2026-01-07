"""
Research Agent - Filters, summarizes, and follows links in search results.

This agent uses a cheaper LLM to:
1. Review raw search results
2. Select only relevant results
3. Identify useful links to crawl for additional data
4. Generate a brief summary for the forecasting agent
"""
import asyncio
import json
import re

from llm import call_llm
from news import exa_search_raw, exa_crawl_urls
from config import RESEARCH_MODEL, RESEARCH_TEMP, RESEARCH_THINKING, GET_NEWS
from prompts import RESEARCH_AGENT_PROMPT, LINK_ANALYSIS_PROMPT


async def run_research_agent(
    question: str, 
    existing_results: list[dict] = None,
    thinking: bool = RESEARCH_THINKING
) -> tuple[list[dict], str]:
    """
    Run the research agent to search, filter, follow links, and summarize.
    
    Args:
        question: The question title/query
        existing_results: Optional list of search results to use (skips fresh search)
    """
    if not GET_NEWS:
        return [], "No research performed (GET_NEWS is disabled)."

    if existing_results:
        print(f"[Research Agent] Using {len(existing_results)} provided search results")
        raw_results = existing_results
    else:
        print(f"[Research Agent] Searching for: {question}")
        raw_results = exa_search_raw(question)
    
    if not raw_results:
        return [], "No search results found."
    
    print(f"[Research Agent] Found {len(raw_results)} results, filtering...")
    
    # Step 2: Filter to relevant results and get summary
    results_json = json.dumps(raw_results, indent=2, default=str)
    
    filter_prompt = RESEARCH_AGENT_PROMPT.format(
        question=question,
        results_json=results_json
    )
    
    filter_response = await call_llm(
        filter_prompt, 
        model=RESEARCH_MODEL, 
        temperature=RESEARCH_TEMP,
        thinking=thinking
    )
    
    relevant_results, summary = parse_research_agent_response(filter_response, raw_results)
    print(f"[Research Agent] Selected {len(relevant_results)} relevant results")
    
    # NOTE: Link crawling disabled - Exa crawl doesn't support date filtering
    # so it could leak future information. Only using search results.
    # crawled_pages = await analyze_and_crawl_links(question, relevant_results)
    # if crawled_pages:
    #     print(f"[Research Agent] Crawled {len(crawled_pages)} additional pages")
    #     for page in crawled_pages:
    #         relevant_results.append({
    #             "title": page.get("title", "Crawled Page"),
    #             "url": page.get("url", ""),
    #             "text": page.get("text", ""),
    #             "published_date": "N/A",
    #             "crawled": True  # Mark as crawled content
    #         })
    
    print(f"[Research Agent] Summary: {summary[:200]}...")
    
    return relevant_results, summary


async def analyze_and_crawl_links(
    question: str, 
    results: list[dict],
    thinking: bool = RESEARCH_THINKING
) -> list[dict]:
    """
    Analyze search results for useful links and crawl the top ones.
    
    Returns:
        list[dict]: Crawled page contents
    """
    if not results:
        return []
    
    # Format the results content for link analysis
    results_content = ""
    for i, result in enumerate(results):
        results_content += (
            f"[Result {i}]:\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"Content: {result.get('text', '')[:1500]}\n\n"
        )
    
    # Ask the agent to identify useful links
    link_prompt = LINK_ANALYSIS_PROMPT.format(
        question=question,
        results_content=results_content
    )
    
    link_response = await call_llm(
        link_prompt,
        model=RESEARCH_MODEL,
        temperature=RESEARCH_TEMP,
        thinking=thinking
    )
    
    # Parse the URLs to crawl
    urls_to_crawl = parse_urls_from_response(link_response)
    
    if not urls_to_crawl:
        print("[Research Agent] No additional links identified for crawling")
        return []
    
    print(f"[Research Agent] Identified {len(urls_to_crawl)} links to crawl: {urls_to_crawl}")
    
    # Crawl the URLs
    crawled = exa_crawl_urls(urls_to_crawl)
    
    return crawled


def parse_urls_from_response(response: str) -> list[str]:
    """
    Extract URLs from the link analysis response.
    
    Expected format:
    URLS_TO_CRAWL: ["url1", "url2", "url3"]
    """
    # Try to find the URLS_TO_CRAWL line
    urls_match = re.search(r'URLS_TO_CRAWL:\s*\[(.*?)\]', response, re.DOTALL)
    
    if not urls_match:
        return []
    
    urls_str = urls_match.group(1)
    
    # Extract quoted strings
    urls = re.findall(r'"([^"]+)"', urls_str)
    
    # Filter to valid URLs (basic check)
    valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
    
    # Limit to 4 URLs max
    return valid_urls[:4]


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
        source_type = "[Crawled]" if result.get('crawled') else "[Search]"
        output += (
            f"{source_type} [{i+1}] {result['title']}\n"
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
