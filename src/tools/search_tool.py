"""
Search Tool - Web search via Exa API, wrapped for tool calling.

This allows the research agent to search for news and context
as part of the tool calling loop.
"""
from datetime import datetime, timedelta
from typing import Optional

from .base import DataTool, ToolResult


class SearchTool(DataTool):
    """
    Web search tool via Exa API.
    
    Searches for news, articles, and context relevant to forecasting questions.
    Can be interleaved with data tools during research.
    """
    
    name = "search_web"
    description = """
Search the web for news, articles, and context relevant to a forecasting question.

Use this to:
- Find recent news about a topic
- Get context for forecasting questions
- Research current events and developments

Returns search results with titles, URLs, and content snippets.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query - be specific and include key entities"
            },
            "num_results": {
                "type": "integer",
                "default": 10,
                "description": "Number of results to return (max 20)"
            },
            "days_back": {
                "type": "integer",
                "description": "Only return results from the last N days (optional)"
            }
        },
        "required": ["query"]
    }
    
    async def execute(
        self,
        query: str,
        num_results: int = 10,
        days_back: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute the web search."""
        try:
            # Import search function
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from news import exa_search_raw
            
            # Calculate date filter if specified
            end_date = None
            if days_back:
                end_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
            # Execute search
            results = exa_search_raw(
                query, 
                num_results=min(num_results, 20),
                end_published_date=end_date
            )
            
            if not results:
                return ToolResult(
                    success=True,
                    data=self.format_data_report(
                        title=f"Search: {query}",
                        data={"results": [], "message": "No results found"},
                        source="Exa Search"
                    ),
                    metadata={"query": query, "num_results": 0}
                )
            
            # Format results
            formatted_results = []
            for i, r in enumerate(results[:num_results]):
                formatted_results.append({
                    "index": i,
                    "title": r.get("title", "N/A"),
                    "url": r.get("url", "N/A"),
                    "published_date": r.get("published_date", "Unknown"),
                    "snippet": r.get("text", "")[:500] + "..." if len(r.get("text", "")) > 500 else r.get("text", "")
                })
            
            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"Search: {query}",
                    data={
                        "query": query,
                        "num_results": len(formatted_results),
                        "results": formatted_results
                    },
                    source="Exa Search"
                ),
                metadata={"query": query, "num_results": len(formatted_results)}
            )
            
        except ImportError:
            return ToolResult(
                success=False,
                data=None,
                error="Exa search not available - check EXA_API_KEY"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Search failed: {str(e)}"
            )
