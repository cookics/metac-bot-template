"""
Tool interface for standardized agent actions.
This enables future expansion with additional tools (Polymarket, Weather, etc.)
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolResult:
    """Standard result from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Tool(ABC):
    """Base class for all agent tools."""
    
    name: str
    description: str
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def to_dict(self) -> dict:
        """Return tool metadata for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
        }


class SearchTool(Tool):
    """Tool for web search via Exa API."""
    
    name = "search"
    description = "Search the web for information relevant to a forecasting question"
    
    def __init__(self):
        from news import exa_search_raw
        self._search_fn = exa_search_raw
    
    async def execute(
        self, 
        query: str, 
        num_results: int = 10,
        end_published_date: str = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute a web search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            end_published_date: Only return results published before this date (ISO format)
        """
        try:
            results = self._search_fn(
                query, 
                num_results=num_results,
                end_published_date=end_published_date
            )
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "query": query,
                    "num_results": len(results),
                    "date_filter": end_published_date,
                    "executed_at": datetime.now().isoformat()
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )


class CrawlTool(Tool):
    """Tool for crawling specific URLs via Exa API."""
    
    name = "crawl"
    description = "Fetch full content from specific URLs"
    
    def __init__(self):
        from news import exa_crawl_urls
        self._crawl_fn = exa_crawl_urls
    
    async def execute(self, urls: list[str], **kwargs) -> ToolResult:
        """
        Crawl specified URLs.
        
        Args:
            urls: List of URLs to crawl
        """
        try:
            results = self._crawl_fn(urls)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "urls_requested": urls,
                    "urls_crawled": len(results),
                    "executed_at": datetime.now().isoformat()
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )


# Registry of available tools
AVAILABLE_TOOLS = {
    "search": SearchTool,
    "crawl": CrawlTool,
}


def get_tool(name: str) -> Tool:
    """Get a tool instance by name."""
    if name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {name}. Available: {list(AVAILABLE_TOOLS.keys())}")
    return AVAILABLE_TOOLS[name]()


def list_tools() -> list[dict]:
    """List all available tools with their metadata."""
    return [get_tool(name).to_dict() for name in AVAILABLE_TOOLS]
