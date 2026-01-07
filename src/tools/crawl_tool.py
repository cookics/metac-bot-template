"""
Web Crawl Tool - Fetch full page content from URLs.

Used when the research agent finds a promising link in search results
and needs the full page content.
"""
from typing import Optional

from .base import DataTool, ToolResult


class CrawlTool(DataTool):
    """
    Web crawl tool via Exa API.
    
    Fetches full page content from URLs found in search results.
    Use when you need more detail than the search snippet provides.
    """
    
    name = "crawl_urls"
    description = """
Fetch full page content from one or more URLs.

Use this when:
- Search results mention something interesting but the snippet isn't enough
- You found a link to a primary source (e.g., government data, official report)
- You need detailed information from a specific page

Returns the full text content of each page (up to 3000 chars each).
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to crawl (max 5)"
            },
            "max_content_length": {
                "type": "integer",
                "default": 3000,
                "description": "Maximum characters to return per page"
            }
        },
        "required": ["urls"]
    }
    
    async def execute(
        self,
        urls: list[str],
        max_content_length: int = 3000,
        **kwargs
    ) -> ToolResult:
        """Execute the web crawl."""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from news import exa_crawl_urls
            
            # Limit to 5 URLs
            urls = urls[:5]
            
            # Execute crawl
            results = exa_crawl_urls(urls)
            
            if not results:
                return ToolResult(
                    success=True,
                    data=self.format_data_report(
                        title=f"Crawl: {len(urls)} URLs",
                        data={"pages": [], "message": "No content retrieved"},
                        source="Exa Crawl"
                    ),
                    metadata={"urls": urls, "pages_retrieved": 0}
                )
            
            # Format results with content length limit
            pages = []
            for r in results:
                text = r.get("text", "")[:max_content_length]
                pages.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": text,
                    "content_length": len(text)
                })
            
            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"Crawled {len(pages)} pages",
                    data={"pages": pages},
                    source="Exa Crawl"
                ),
                metadata={"urls": urls, "pages_retrieved": len(pages)}
            )
            
        except ImportError:
            return ToolResult(
                success=False,
                data=None,
                error="Exa crawl not available - check EXA_API_KEY"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Crawl failed: {str(e)}"
            )
