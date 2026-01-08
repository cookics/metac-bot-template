import asyncio
import requests
from datetime import datetime
from typing import Optional

from ..base import DataTool, ToolResult


class ManifoldMarketsTool(DataTool):
    """
    Manifold Markets data fetching tool.
    
    Searches for high-signal prediction markets on Manifold.
    """
    
    name = "search_manifold"
    description = """
Search for high-signal prediction markets on Manifold Markets.
Retrieves probabilities for binary markets and top outcomes for multi-choice markets.

Use this for:
- Getting market sentiment on specific events or terms
- Finding high-liquidity prediction markets
- Analyzing competitive outcomes (multi-choice)
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "The search term or topic (e.g., 'Federal Reserve', 'Nvidia', 'Election')"
            },
            "min_volume": {
                "type": "integer",
                "default": 500,
                "description": "Minimum volume to filter high-signal markets"
            },
            "min_bettors": {
                "type": "integer",
                "default": 10,
                "description": "Minimum number of unique bettors"
            },
            "limit": {
                "type": "integer",
                "default": 50,
                "description": "Maximum number of markets to return"
            }
        },
        "required": ["term"]
    }
    
    async def execute(
        self,
        term: str,
        min_volume: int = 500,
        min_bettors: int = 10,
        limit: int = 50
    ) -> ToolResult:
        """Execute the Manifold Markets search."""
        try:
            base_search_url = "https://api.manifold.markets/v0/search-markets"
            base_market_url = "https://api.manifold.markets/v0/market"
            headers = {"User-Agent": "manifold-fed-quant/1.1"}
            
            params = {
                "term": term,
                "sort": "liquidity",
                "filter": "open",
                "limit": limit
            }
            
            # Run in a thread since requests is synchronous
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, 
                lambda: requests.get(base_search_url, params=params, headers=headers, timeout=10)
            )
            resp.raise_for_status()
            markets = resp.json()
            
            if not markets:
                return ToolResult(
                    success=True,
                    data=self.format_data_report(
                        title=f"Manifold Markets - {term}",
                        data={"markets": [], "counts": 0},
                        source="Manifold Markets"
                    ),
                    metadata={"term": term, "count": 0}
                )

            all_rows = []
            min_prob = 0.01  # Skip noise

            for m in markets:
                vol = m.get("volume", 0)
                bettors = m.get("uniqueBettorCount", 0)
                
                if vol < min_volume or bettors < min_bettors:
                    continue

                base = {
                    "question": m["question"],
                    "volume": round(vol, 0),
                    "bettors": bettors,
                    "type": m["outcomeType"],
                    "url": f"https://manifold.markets/{m['creatorUsername']}/{m['slug']}"
                }

                # BINARY
                if m["outcomeType"] == "BINARY":
                    prob = m.get("probability")
                    if prob:
                        all_rows.append({**base, "answer": "YES", "prob": round(prob, 4)})

                # MULTI-CHOICE / FREE-RESPONSE
                elif m["outcomeType"] in {"FREE_RESPONSE", "MULTIPLE_CHOICE"}:
                    # Fetch answers
                    a_resp = await loop.run_in_executor(
                        None,
                        lambda: requests.get(f"{base_market_url}/{m['id']}/answers", headers=headers, timeout=10)
                    )
                    if a_resp.status_code != 200:
                        continue
                    
                    answers = a_resp.json()
                    valid_answers = []
                    for a in answers:
                        # API uses different keys sometimes
                        p = a.get("probability") or a.get("prob") or a.get("cpmmProbability")
                        if p and p >= min_prob:
                            valid_answers.append({"text": a.get("text"), "prob": p})
                    
                    # Sort by prob descending and take top 5
                    top_answers = sorted(valid_answers, key=lambda x: x['prob'], reverse=True)[:5]
                    
                    for a in top_answers:
                        all_rows.append({**base, "answer": a['text'], "prob": round(a['prob'], 4)})
                    
                    await asyncio.sleep(0.1)  # Polite delay

            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"Manifold Markets - {term}",
                    data={
                        "markets": all_rows,
                        "count": len(all_rows)
                    },
                    source="Manifold Markets"
                ),
                metadata={"term": term, "count": len(all_rows)}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Manifold Markets search failed: {str(e)}"
            )
