"""
PolyMarket Tool - Search for active events and markets on PolyMarket.
"""
import requests
import json
from typing import List, Optional
from datetime import datetime

from ..base import DataTool, ToolResult

class PolyMarketSearchTool(DataTool):
    """
    Search for active markets on PolyMarket via Gamma API.
    
    Returns top markets by volume matching the search terms.
    """
    
    name = "search_polymarket"
    description = """
Search PolyMarket for active events and markets matching specific terms.

Use this to:
- Get current market probabilities for specific events
- Find active prediction markets related to a research topic
- Compare market sentiment across different outcomes

Returns a list of markets with their questions, current probabilities, and volume.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "search_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search terms or keywords to filter markets (e.g. ['Trump', 'Election'])"
            },
            "limit": {
                "type": "integer",
                "default": 30,
                "description": "Number of top markets to return"
            }
        },
        "required": ["search_terms"]
    }
    
    async def execute(
        self,
        search_terms: List[str],
        limit: int = 30,
        **kwargs
    ) -> ToolResult:
        """Execute the PolyMarket search."""
        url = "https://gamma-api.polymarket.com/events"
        params = {
            "limit": 100, # Fetch a decent batch to filter
            "closed": "false"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            events = response.json()
            
            detailed_markets = []
            
            # Filter terms - case insensitive
            terms = [t.lower() for t in search_terms]
            
            for event in events:
                title = event.get('title', '')
                volume = float(event.get('volume', 0))
                
                # Check if event title matches any search term
                title_match = any(term in title.lower() for term in terms)
                
                markets = event.get('markets', [])
                for market in markets:
                    question = market.get('question', title)
                    
                    # Check if market question matches any search term
                    question_match = any(term in question.lower() for term in terms)
                    
                    if not (title_match or question_match):
                        continue
                        
                    outcomes = market.get('outcomes', [])
                    outcome_prices = market.get('outcomePrices', [])
                    
                    # Parse outcome prices and outcomes if they are strings
                    if isinstance(outcome_prices, str):
                        try:
                            outcome_prices = json.loads(outcome_prices)
                        except:
                            outcome_prices = []
                            
                    if isinstance(outcomes, str):
                        try:
                            outcomes = json.loads(outcomes)
                        except:
                            outcomes = []

                    # Pair outcomes with prices
                    probs_str = []
                    probs_map = {}
                    if len(outcomes) == len(outcome_prices):
                        for out, price in zip(outcomes, outcome_prices):
                            try:
                                p_val = float(price)
                                probs_str.append(f"{out}: {p_val:.1%}")
                                probs_map[out] = p_val
                            except:
                                pass
                    
                    detailed_markets.append({
                        'event': title,
                        'question': question,
                        'probabilities': ", ".join(probs_str),
                        'probs_raw': probs_map,
                        'volume': volume,
                        'market_id': market.get('id'),
                        'url': f"https://polymarket.com/event/{event.get('slug')}"
                    })

            # Sort by volume and limit
            detailed_markets.sort(key=lambda x: x['volume'], reverse=True)
            results = detailed_markets[:limit]
            
            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"PolyMarket Search: {', '.join(search_terms)}",
                    source="PolyMarket Gamma API",
                    data={
                        "search_terms": search_terms,
                        "count": len(results),
                        "markets": results
                    }
                ),
                metadata={"search_terms": search_terms, "results_count": len(results)}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"PolyMarket search failed: {str(e)}"
            )
