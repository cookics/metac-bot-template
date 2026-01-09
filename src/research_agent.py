"""
Research Agent - Filters, summarizes, and follows links in search results.

This agent uses a cheaper LLM to:
1. Review raw search results
2. Select only relevant results
3. Identify useful links to crawl for additional data
4. Generate a brief summary for the forecasting agent
5. (NEW) Call specialized tools for market data and forecasts
"""
import asyncio
import json
import re

from llm import call_llm
from news import exa_search_raw, exa_crawl_urls
from config import RESEARCH_MODEL, RESEARCH_TEMP, RESEARCH_THINKING, GET_NEWS

# Import standard prompts
from prompts import RESEARCH_AGENT_PROMPT, LINK_ANALYSIS_PROMPT

# Tool calling imports
try:
    from tools import get_all_tools, get_market_tools, get_data_tools, get_research_tools
    from tools.executor import run_tool_calling_loop
    from tools.formatting import (
        format_for_forecaster, 
        format_tool_results_compact,
        extract_percentiles_from_tool
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    print("[Research Agent] Warning: Tool calling infrastructure not available")


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
    Optimized to be more compact by default.
    """
    output = f"=== RESEARCH SYNTHESIS ===\n{summary}\n\n"
    output += "=== SOURCES REFERENCED ===\n"
    
    for i, result in enumerate(relevant_results[:10]):
        source_type = "[Crawled]" if result.get('crawled') else "[Search]"
        output += (
            f"{source_type} [{i+1}] {result['title']}\n"
            f"    URL: {result['url']}\n"
            f"    Published: {result.get('published_date', 'Unknown')}\n"
        )
    
    return output


# ========================= TOOL-CALLING RESEARCH =========================

# System prompt for tool-calling research agent
TOOL_RESEARCH_SYSTEM_PROMPT = """Be concise. The forecaster will use your report to make predictions."""


# Dedicated synthesis prompt for creating an authoritative, deep factual report
SYNTHESIS_STEP_PROMPT = """You are a Lead Intelligence Analyst. You have been given a collection of raw search results, news snippets, and specialized tool data regarding a specific forecasting question.

Your task is to write a COMPREHENSIVE, AUTHORITATIVE, and FACT-DENSE Research Report that will be used by a Superforecaster to make a final prediction.

Question to Synthesis:
{question}

=== INPUT DATA ===
{raw_data}

=== INSTRUCTIONS ===
1. **Depth over Brevity**: Write a detailed report (targeting 1500-3000 tokens). Do not be afraid of length if it adds factual depth.
2. **Thematic Organization**: Organize by causal drivers, market themes, or sectors—NOT by source.
3. **Evidence-Based**: Every claim must be backed by data found in the input. Cite specific numbers, dates, and names.
4. **Synthesis, not Summary**: Do not simply list what each source said. Instead, integrate the information to show how different facts interact.
5. **Quantitative Focus**: Prioritize hard data, market prices, percentiles, and statistical historical base rates.
6. **Causal Reasoning**: Explicitly identify the mechanisms that will drive the outcome.
7. **Authoritative Tone**: Write like you are briefing a head of state or a top-tier hedge fund manager.

Your goal is to "do the thinking" and "data-mining" for the forecaster, so they can focus entirely on probabilistic calibration. Spew ALL important facts into this report.

REPORT STRUCTURE:
- Executive Summary
- Current Market/Situational Context
- Deep Dive: Key Causal Drivers
- Historical Base Rates & Comparative Situations
- Counter-Arguments & Potential Surprises
- Factual Data Appendix (if needed for raw lists of numbers)
"""


async def run_research_with_tools(
    question: str,
    question_type: str = "general",
    use_all_tools: bool = False
) -> tuple[str, list[dict]]:
    """
    Run research agent with tool calling capability.
    
    Args:
        question: The forecasting question
        question_type: "market" for market forecast questions, "general" otherwise
        use_all_tools: If True, provide all tools. If False, provide relevant subset.
    
    Returns:
        Tuple of (research_report, tool_calls_made)
    """
    if not TOOLS_AVAILABLE:
        print("[Research Agent] Tool calling not available, falling back to standard research")
        results, summary = await run_research_agent(question)
        return format_results_for_forecaster(results, summary), []
    
    # Select relevant tools
    if use_all_tools:
        tools = get_all_tools()
    elif question_type == "market":
        tools = get_market_tools() + get_data_tools()
    else:
        tools = get_data_tools()
    
    print(f"[Research Agent] Running with {len(tools)} tools: {[t.name for t in tools]}")
    
    # Build the research prompt
    research_prompt = f"""Question to forecast:
{question}

Use the available tools to gather relevant data and forecasts.
Then write a SHORT REPORT with your findings.

If this is a market question (Treasury yields, stock spreads, VIX, etc.), 
use the appropriate forecast_* tool to get probability distributions.

After tool calls, synthesize your findings into a clear research report.
"""
    
    # Run the tool calling loop
    final_response, tool_calls, messages = await run_tool_calling_loop(
        initial_prompt=research_prompt,
        tools=tools,
        model=RESEARCH_MODEL,
        temperature=RESEARCH_TEMP,
        max_iterations=5,
        system_prompt=TOOL_RESEARCH_SYSTEM_PROMPT
    )
    
    print(f"[Research Agent] Tool calling complete. Made {len(tool_calls)} tool calls.")
    
    # Format tool results for the forecaster
    report = "=== RESEARCH REPORT (Tool-Assisted) ===\n\n"
    
    # Add tool call summaries
    if tool_calls:
        report += "DATA SOURCES:\n"
        for tc in tool_calls:
            report += f"  - {tc['tool_name']}: "
            if tc.get('error'):
                report += f"ERROR: {tc['error']}\n"
            else:
                report += "SUCCESS\n"
        report += "\n"
    
    # Add the model's synthesis
    report += "SYNTHESIS:\n"
    report += final_response
    
    return report, tool_calls, messages


def format_tool_results_for_forecast(tool_calls: list[dict]) -> str:
    """
    Format tool results specifically for numeric/market forecast questions.
    Extracts percentile data when available.
    """
    output = ""
    
    for tc in tool_calls:
        if tc.get('error') or not tc.get('result'):
            continue
        
        result = tc['result']
        
        # Check if this is a forecast distribution result
        if isinstance(result, dict) and result.get('type') == 'forecast_distribution':
            output += f"\n=== {tc['tool_name'].upper()} DISTRIBUTION ===\n"
            output += f"Question ID: {result.get('question_id', 'N/A')}\n"
            
            if 'percentiles' in result:
                p = result['percentiles']
                output += f"Percentiles:\n"
                output += f"  1st: {p.get('p1', 'N/A')}\n"
                output += f"  5th: {p.get('p5', 'N/A')}\n"
                output += f"  25th: {p.get('p25', 'N/A')}\n"
                output += f"  50th (Median): {p.get('p50', 'N/A')}\n"
                output += f"  75th: {p.get('p75', 'N/A')}\n"
                output += f"  95th: {p.get('p95', 'N/A')}\n"
                output += f"  99th: {p.get('p99', 'N/A')}\n"
            
            if 'statistics' in result:
                s = result['statistics']
                output += f"Statistics:\n"
                output += f"  Mean: {s.get('mean', 'N/A')}\n"
                output += f"  Std: {s.get('std', 'N/A')}\n"
            
            if 'assumptions' in result:
                output += f"Assumptions:\n"
                for k, v in result['assumptions'].items():
                    output += f"  {k}: {v}\n"
            
            output += "\n"
        
        # Check for tail probabilities (VIX specific)
        if isinstance(result, dict) and 'tail_probabilities' in result:
            output += "Tail Probabilities:\n"
            for k, v in result['tail_probabilities'].items():
                output += f"  {k}: {v:.1%}\n"
            output += "\n"
    
    return output


# Backward-compatible wrapper for existing code
def run_research(question: str, use_tools: bool = False) -> str:
    """
    Wrapper that runs the research agent and returns formatted results.
    This maintains backward compatibility with existing forecasting code.
    
    Args:
        question: The forecasting question
        use_tools: If True, use tool-calling research agent
    """
    if use_tools and TOOLS_AVAILABLE:
        report, tool_calls, messages = asyncio.run(run_research_with_tools(question))
        
        # If we got forecast distributions, append the formatted version
        if tool_calls:
            tool_output = format_tool_results_for_forecast(tool_calls)
            if tool_output:
                report += "\n\n=== RAW TOOL DATA ===\n" + tool_output
        
        return report
    else:
        relevant_results, summary = asyncio.run(run_research_agent(question))
        return format_results_for_forecaster(relevant_results, summary)


# ========================= UNIFIED PIPELINE =========================

async def run_research_pipeline(
    question: str,
    question_type: str = "general"
) -> dict:
    """
    Unified research pipeline with tool calling.
    
    Flow:
    1. Initial web search (always, unless market-only question)
    2. Tool calling loop (model decides: crawl, data, forecast)
    3. Return structured results (not regenerated text)
    
    Args:
        question: The forecasting question
        question_type: "market" for market-specific, "general" otherwise
    
    Returns:
        dict with:
        - synthesis: Model's written summary
        - tool_calls: Raw tool call results (structured)
        - search_results: Top search hits
        - percentiles: Extracted percentiles if from forecast tools (optional)
        - formatted_for_forecaster: Compact string ready to pass to forecaster
    """
    if not TOOLS_AVAILABLE:
        # Fallback to standard research
        results, summary = await run_research_agent(question)
        return {
            "synthesis": summary,
            "tool_calls": [],
            "search_results": results,
            "percentiles": None,
            "formatted_for_forecaster": format_results_for_forecaster(results, summary),
            "messages": []
        }
    
    print(f"[Research Pipeline] Starting for: {question[:60]}...")
    
    # Step 1: Initial web search (with cost tracking)
    search_results = []
    exa_cost = 0.0
    if question_type != "market":
        # For general questions, always search first
        print("[Research Pipeline] Step 1: Web search")
        search_result = exa_search_raw(question, num_results=10, return_cost=True)
        if isinstance(search_result, tuple):
            search_results, cost_info = search_result
            exa_cost += cost_info.get("total", 0.0)
            print(f"[Research Pipeline] Found {len(search_results)} search results (Exa cost: ${exa_cost:.4f})")
        else:
            search_results = search_result
            print(f"[Research Pipeline] Found {len(search_results)} search results")
    
    # Step 2: Select tools based on question type
    if question_type == "market":
        # Market questions: forecast tools + data tools
        tools = get_market_tools() + get_data_tools()
    else:
        # General questions: research tools (search, crawl, data)
        tools = get_research_tools()
    
    print(f"[Research Pipeline] Step 2: Tool loop with {len(tools)} tools")
    
    # Build prompt with search context if available
    if search_results:
        from tools.formatting import format_search_results_compact
        search_context = format_search_results_compact(search_results, max_results=5)
        research_prompt = f"""Question to forecast:
{question}

Initial search results:
{search_context}

Based on these search results and your available tools:
1. If you see an interesting link, use crawl_urls to get full content
2. If you need financial data, use the data tools (yahoo, fred, options)
3. If this is a market question, use forecast tools for distributions

When done, write a PRELIMINARY summary. After this, I will ask you for a deeper synthesis.
"""
    else:
        research_prompt = f"""Question to forecast:
{question}

This appears to be a market question. Use the forecast tools to generate
probability distributions, and data tools to get current values.

When done, write a PRELIMINARY summary. After this, I will ask you for a deeper synthesis.
"""
    
    # Run tool calling loop
    final_response, tool_calls, messages = await run_tool_calling_loop(
        initial_prompt=research_prompt,
        tools=tools,
        model=RESEARCH_MODEL,
        temperature=RESEARCH_TEMP,
        max_iterations=2,  # Reduced from 5 - model should get data in 1-2 rounds
        system_prompt=TOOL_RESEARCH_SYSTEM_PROMPT
    )
    
    print(f"[Research Pipeline] Step 3: Made {len(tool_calls)} tool calls")
    
    # Step 4: NEW - Dedicated Synthesis Step
    print("[Research Pipeline] Step 4: Running high-quality synthesis...")
    
    # Prepare all raw data for the synthesis prompt
    from tools.formatting import format_search_results_full, format_tool_results_full
    raw_data_block = ""
    if search_results:
        raw_data_block += "=== WEB SEARCH RESULTS ===\n" + format_search_results_full(search_results, max_results=10) + "\n\n"
    if tool_calls:
        raw_data_block += "=== TOOL DATA RESULTS ===\n" + format_tool_results_full(tool_calls) + "\n\n"
        
    synthesis_prompt = SYNTHESIS_STEP_PROMPT.format(
        question=question,
        raw_data=raw_data_block
    )
    
    # Call Grok for the deep synthesis
    deep_synthesis = await call_llm(
        synthesis_prompt,
        model=RESEARCH_MODEL,
        temperature=0.4, # Lower temp for factual synthesis
        thinking=True # Thinking is very helpful for cross-referencing data
    )
    
    # Count tool usage
    tool_usage = {}
    for tc in tool_calls:
        name = tc.get("tool_name", "unknown")
        tool_usage[name] = tool_usage.get(name, 0) + 1
    
    # Step 5: Extract percentiles if available (for direct CDF use)
    percentiles = extract_percentiles_from_tool(tool_calls) if TOOLS_AVAILABLE else None
    
    # Step 6: Format OPTIMIZED output for forecaster
    # This uses the deep_synthesis as the main body and strips raw snippets
    from tools.formatting import format_for_forecaster_optimized
    formatted = format_for_forecaster_optimized(
        research_synthesis=deep_synthesis,
        search_results=search_results,
        tool_calls=tool_calls
    )
    
    print(f"[Research Pipeline] Complete. Synthesis length: {len(deep_synthesis)} chars. Final prompt size: {len(formatted)} chars")
    
    return {
        "synthesis": deep_synthesis,
        "tool_calls": tool_calls,
        "search_results": search_results,
        "percentiles": percentiles,
        "messages": messages,
        "formatted_for_forecaster": formatted,
        "exa_cost": exa_cost,
        "tool_usage": tool_usage
    }


def run_research_sync(question: str, question_type: str = "general") -> dict:
    """
    Synchronous wrapper for run_research_pipeline.
    
    Returns the full pipeline result dict.
    """
    return asyncio.run(run_research_pipeline(question, question_type))

