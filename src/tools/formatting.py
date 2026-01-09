"""
Formatting for Tool Results - Full Data for Forecaster.

These functions format tool outputs for the forecaster model.
Key principle: Give the forecaster COMPLETE data, not summaries.

For percentiles: Include ALL of them
For search: Include full snippets of relevant articles
For data: Include raw JSON, not summarized stats
"""
import json
from typing import Any


def format_search_results_full(results: list[dict], max_results: int = 5) -> str:
    """
    Format search results for the forecaster.
    Truncates snippets to save tokens.
    """
    if not results:
        return "[No search results]"
    
    lines = []
    for i, r in enumerate(results[:max_results]):
        title = r.get("title", "Untitled")
        date = r.get("published_date", "Unknown")
        url = r.get("url", "")
        # Reduced from 800 to 300 to save tokens
        text = r.get("text", "")[:300]
        
        lines.append(f"""[Source {i+1}]
Title: {title}
Date: {date}
URL: {url}
Content: {text}
""")
    
    return "\n".join(lines)


def format_tool_results_full(tool_calls: list[dict]) -> str:
    """
    Format tool results with FULL data for the forecaster.
    
    For forecast tools: ALL percentiles as JSON
    For data tools: Full data dict
    """
    if not tool_calls:
        return "[No tool data]"
    
    sections = []
    
    for tc in tool_calls:
        if tc.get("error"):
            sections.append(f"[{tc['tool_name']}] ERROR: {tc['error']}")
            continue
        
        result = tc.get("result")
        if not result:
            continue
        
        tool_name = tc["tool_name"]
        
        # Forecast tools - give FULL percentiles
        if tool_name.startswith("forecast_"):
            formatted = _format_forecast_full(tool_name, result)
            if formatted:
                sections.append(formatted)
        
        # Data tools - give full data
        elif tool_name in ["get_yahoo_data", "get_fred_data", "get_options_data", "search_manifold"]:
            formatted = _format_data_full(tool_name, result)
            if formatted:
                sections.append(formatted)
        
        # Search/crawl results
        elif tool_name in ["search_web", "crawl_urls"]:
            formatted = _format_search_crawl_full(tool_name, result)
            if formatted:
                sections.append(formatted)
    
    return "\n\n".join(sections) if sections else "[No tool data]"


def _format_forecast_full(tool_name: str, result: Any) -> str:
    """Format forecast tool output with FULL percentiles."""
    if not isinstance(result, dict):
        return ""
    
    # Handle nested structures
    if "treasury_10y_yield" in result:
        parts = []
        parts.append(_format_single_forecast_full("10Y TREASURY YIELD", result["treasury_10y_yield"]))
        if "hy_oas" in result:
            parts.append(_format_single_forecast_full("HIGH YIELD OAS", result["hy_oas"]))
        return "\n\n".join(parts)
    elif "hy_oas" in result:
        return _format_single_forecast_full("HIGH YIELD OAS", result["hy_oas"])
    elif result.get("type") == "forecast_distribution":
        return _format_single_forecast_full(tool_name.upper(), result)
    
    return ""


def _format_single_forecast_full(name: str, data: dict) -> str:
    """Format a single forecast with ALL percentiles."""
    lines = [f"=== {name} FORECAST ==="]
    
    # Full percentiles as formatted list
    if "percentiles" in data:
        lines.append("PERCENTILES:")
        p = data["percentiles"]
        # Standard percentiles to always show
        for pct in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98, 99]:
            key = f"p{pct}"
            if key in p:
                val = p[key]
                lines.append(f"  {pct}th: {val:.4f}" if isinstance(val, float) else f"  {pct}th: {val}")
    
    # Statistics
    if "statistics" in data:
        s = data["statistics"]
        lines.append("STATISTICS:")
        for key in ["mean", "std", "min", "max", "n_samples"]:
            if key in s:
                val = s[key]
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.4f}")
                else:
                    lines.append(f"  {key}: {val}")
    
    # Tail probabilities (VIX)
    if "tail_probabilities" in data:
        lines.append("TAIL PROBABILITIES:")
        for k, v in data["tail_probabilities"].items():
            lines.append(f"  {k}: {v:.2%}")
    
    # Assumptions
    if "assumptions" in data:
        lines.append("ASSUMPTIONS:")
        for k, v in data["assumptions"].items():
            lines.append(f"  {k}: {v}")
    
    return "\n".join(lines)


def _format_data_full(tool_name: str, result: Any) -> str:
    """Format data tool output with full data."""
    if not isinstance(result, dict):
        return ""
    
    if tool_name == "search_manifold":
        return _format_manifold_markets(result)
    
    title = result.get("title", tool_name)
    data = result.get("data", result)
    
    lines = [f"=== {title.upper()} ==="]
    
    # Just pretty-print the data dict
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ["type", "fetched_at"]:
                continue  # Skip metadata
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k2, v2 in value.items():
                    if isinstance(v2, float):
                        lines.append(f"  {k2}: {v2:.4f}")
                    else:
                        lines.append(f"  {k2}: {v2}")
            elif isinstance(value, list):
                lines.append(f"{key}: {value[:5]}...")  # First 5 items
            elif isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
    
    return "\n".join(lines)


def _format_manifold_markets(result: dict) -> str:
    """Format Manifold Markets search results."""
    data = result.get("data", {})
    markets = data.get("markets", [])
    
    if not markets:
        return "[No high-signal Manifold markets found]"
    
    lines = ["=== MANIFOLD MARKETS REPORT ==="]
    
    # Group by question to avoid repeats if any
    for i, m in enumerate(markets[:15]): # Limit to top 15 rows
        lines.append(f"\n[{i+1}] {m['question']}")
        lines.append(f"    Answer: {m['answer']} | Prob: {m['prob']:.2%}")
        lines.append(f"    Volume: ${m['volume']:,.0f} | Bettors: {m['bettors']}")
        lines.append(f"    Type: {m['type']} | URL: {m['url']}")
        
    return "\n".join(lines)


def _format_search_crawl_full(tool_name: str, result: Any) -> str:
    """Format search/crawl results with strict token limits."""
    if not isinstance(result, dict):
        return ""
    
    data = result.get("data", result)
    
    lines = [f"=== {tool_name.upper()} RESULTS ==="]
    
    if "results" in data:
        # Reduced from 8 to 5
        for r in data["results"][:5]:
            lines.append(f"\n[{r.get('title', 'Untitled')}]")
            lines.append(f"Date: {r.get('published_date', 'Unknown')}")
            lines.append(f"URL: {r.get('url', '')}")
            # Reduced from 600 to 300
            snippet = r.get("snippet", r.get("content", ""))[:300]
            lines.append(f"Content: {snippet}")
    
    if "pages" in data:
        # Limit to 2 pages max in the report to save tokens
        for p in data["pages"][:2]:
            lines.append(f"\n[{p.get('title', 'Page')}]")
            lines.append(f"URL: {p.get('url', '')}")
            # Reduced from 1500 to 500
            content = p.get("content", "")[:500]
            lines.append(f"Content: {content}")
    
    return "\n".join(lines)


def extract_percentiles_from_tool(tool_calls: list[dict]) -> dict | None:
    """
    Extract percentile values from market forecast tools.
    Returns dict with numeric percentile keys (1, 5, 10, ..., 99).
    """
    for tc in tool_calls:
        if tc.get("error") or not tc.get("result"):
            continue
        
        result = tc["result"]
        percentiles = None
        
        if isinstance(result, dict):
            if "percentiles" in result:
                percentiles = result["percentiles"]
            elif "treasury_10y_yield" in result:
                percentiles = result["treasury_10y_yield"].get("percentiles")
            elif "hy_oas" in result:
                percentiles = result["hy_oas"].get("percentiles")
        
        if percentiles:
            numeric_percentiles = {}
            for k, v in percentiles.items():
                if k.startswith("p"):
                    try:
                        numeric_percentiles[int(k[1:])] = float(v)
                    except (ValueError, TypeError):
                        pass
            
            if numeric_percentiles:
                return numeric_percentiles
    
    return None


def format_for_forecaster_optimized(
    research_synthesis: str,
    search_results: list[dict],
    tool_calls: list[dict]
) -> str:
    """
    Optimized formatting for the forecaster.
    - Priortizes the Deep Synthesis (Grok).
    - Minimizes raw data spew (no raw text snippets if they are in the synthesis).
    - Keeps critical numbers/percentiles.
    """
    sections = []
    
    # 1. Grok's Deep Synthesis (The main course)
    if research_synthesis:
        sections.append("=== RESEARCH SYNTHESIS (Lead Analyst Report) ===\n" + research_synthesis)
    
    # 2. Critical Quantitative Data (Percentiles, Stats)
    quantitative_data = _format_critical_data_only(tool_calls)
    if quantitative_data:
        sections.append("=== CRITICAL QUANTITATIVE DATA ===\n" + quantitative_data)
    
    # 3. Source List (Reference only, no text snippets)
    if search_results:
        source_list = "=== SOURCES REFERENCED ===\n"
        for i, r in enumerate(search_results[:10]):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            date = r.get("published_date", "Unknown")
            source_list += f"[{i+1}] {title} ({date})\n    URL: {url}\n"
        sections.append(source_list)
        
    return "\n\n".join(sections)


def _format_critical_data_only(tool_calls: list[dict]) -> str:
    """Extract only the most important numbers from tool calls, avoiding long text."""
    if not tool_calls:
        return ""
    
    parts = []
    for tc in tool_calls:
        if tc.get("error"): continue
        result = tc.get("result")
        if not result: continue
        
        tool_name = tc["tool_name"]
        
        # Keep Market Forecasts (Percentiles)
        if tool_name.startswith("forecast_") or tool_name == "get_options_data":
            parts.append(_format_single_forecast_full(tool_name.upper(), result))
            
        # Keep Manifold Probs (Binary)
        elif tool_name == "search_manifold":
            markets = result.get("data", {}).get("markets", [])
            if markets:
                m_lines = ["Manifold Market Odds:"]
                for m in markets[:5]:
                    m_lines.append(f"  - {m['question']}: {m['prob']:.1%}")
                parts.append("\n".join(m_lines))
                
        # Keep FRED/Yahoo Latest Value
        elif tool_name in ["get_yahoo_data", "get_fred_data"]:
            data = result.get("data", result)
            if isinstance(data, dict):
                p_lines = [f"{tool_name.upper()} Key Stats:"]
                # Only keep top-level numeric values
                for k, v in data.items():
                    if isinstance(v, (int, float)) and k not in ["type", "fetched_at"]:
                        p_lines.append(f"  - {k}: {v}")
                if len(p_lines) > 1:
                    parts.append("\n".join(p_lines))
                    
    return "\n\n".join(parts)


def format_for_forecaster(
    research_synthesis: str,
    search_results: list[dict],
    tool_calls: list[dict]
) -> str:
    """
    Format all research data for the forecaster prompt.
    Gives COMPLETE data - full percentiles, full search results.
    """
    sections = []
    if research_synthesis:
        sections.append("=== RESEARCH SYNTHESIS ===\n" + research_synthesis)
    if search_results:
        sections.append("=== SEARCH RESULTS ===\n" + format_search_results_full(search_results))
    tool_data = format_tool_results_full(tool_calls)
    if tool_data and tool_data != "[No tool data]":
        sections.append(tool_data)
    return "\n\n".join(sections)
format_search_results_compact = format_search_results_full
format_tool_results_compact = format_tool_results_full
