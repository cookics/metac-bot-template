"""
Google Trends Tool - Fetch search interest data via pytrends.
"""
import time
from datetime import datetime, timedelta
from typing import List, Optional

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

from ..base import DataTool, ToolResult

class GoogleTrendsTool(DataTool):
    """
    Fetch Google Trends interest data for given keywords.
    
    Returns interest over time (0-100) for the specified keywords and timeframe.
    """
    
    name = "get_google_trends"
    description = """
Fetch Google Trends interest data for one or more keywords.

Use this to:
- Identify rising search interest in a topic
- Compare the popularity of different entities or terms over time
- Get a proxy for public attention or sentiment change

Returns a report with interest over time data points (0-100 normalized).
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of keywords to search for (max 5)"
            },
            "days_back": {
                "type": "integer",
                "default": 30,
                "description": "Number of days of historical data to fetch"
            },
            "geo": {
                "type": "string",
                "default": "US",
                "description": "Two-letter country code (e.g., 'US', 'GB') or empty for worldwide"
            }
        },
        "required": ["keywords"]
    }
    
    async def execute(
        self,
        keywords: List[str],
        days_back: int = 30,
        geo: str = "US",
        **kwargs
    ) -> ToolResult:
        """Execute the Google Trends search."""
        if not PYTRENDS_AVAILABLE:
            return ToolResult(
                success=False,
                data=None,
                error="pytrends library not installed. Please install it with 'pip install pytrends'."
            )
            
        try:
            # Initialize pytrends
            pytrends = TrendReq(hl="en-US", tz=360)
            
            # Calculate timeframe
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            timeframe_str = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
            
            # Build payload
            pytrends.build_payload(
                keywords[:5], # Max 5 keywords
                timeframe=timeframe_str,
                geo=geo
            )
            
            # Wait a moment to be polite
            time.sleep(1)
            
            # Get interest over time
            df = pytrends.interest_over_time()
            
            if df.empty:
                return ToolResult(
                    success=True,
                    data=self.format_data_report(
                        title=f"Google Trends: {', '.join(keywords)}",
                        source="Google Trends via pytrends",
                        data={"message": "No data found for these keywords/timeframe"}
                    ),
                    metadata={"keywords": keywords, "days_back": days_back}
                )
            
            # Convert dataframe to a formatable dict
            # We'll take the most recent points and some summary stats
            results = []
            for date, row in df.iterrows():
                date_str = date.strftime("%Y-%m-%d")
                point = {"date": date_str}
                for kw in keywords[:5]:
                    if kw in row:
                        point[kw] = int(row[kw])
                results.append(point)
            
            # Summary statistics
            stats = {}
            for kw in keywords[:5]:
                if kw in df.columns:
                    stats[kw] = {
                        "average": float(df[kw].mean()),
                        "max": int(df[kw].max()),
                        "current": int(df[kw].iloc[-1]) if not df.empty else 0
                    }
            
            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"Google Trends: {', '.join(keywords)}",
                    source="Google Trends via pytrends",
                    data={
                        "timeframe": timeframe_str,
                        "geo": geo,
                        "statistics": stats,
                        "recent_data": results[-14:] # Last 2 weeks of data
                    }
                ),
                metadata={"keywords": keywords, "results_count": len(results)}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Google Trends failed: {str(e)}"
            )
