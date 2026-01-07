"""
Options Data Tool.

Fetches options chain data, implied volatility, and expiration info.
"""
import numpy as np
from datetime import datetime, date
from typing import Optional

from ..base import DataTool, ToolResult


class OptionsDataTool(DataTool):
    """
    Options chain and implied volatility data tool.
    
    Returns options data including IV, chain details, and expirations.
    """
    
    name = "get_options_data"
    description = """
Fetch options chain data and extract implied volatility for a ticker.

Capabilities:
- Get ATM (at-the-money) implied volatility
- Fetch full options chain for a specific expiration
- List available expiration dates
- Get IV for puts or calls separately

Use this for:
- Understanding market-implied volatility
- Options pricing context
- Tail risk assessment (put IV vs call IV)
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Underlying ticker symbol (e.g., SPY, AAPL, HYG)"
            },
            "expiration": {
                "type": "string",
                "description": "Specific expiration date (YYYY-MM-DD) or 'nearest' for closest"
            },
            "data_type": {
                "type": "string",
                "enum": ["atm_iv", "chain_summary", "expirations", "iv_skew"],
                "default": "atm_iv",
                "description": "Type of data: 'atm_iv' for ATM implied vol, 'chain_summary' for chain stats, 'expirations' for available dates, 'iv_skew' for put/call IV comparison"
            },
            "option_type": {
                "type": "string",
                "enum": ["calls", "puts", "both"],
                "default": "both",
                "description": "Which options to analyze"
            }
        },
        "required": ["ticker"]
    }
    
    async def execute(
        self,
        ticker: str,
        expiration: Optional[str] = None,
        data_type: str = "atm_iv",
        option_type: str = "both"
    ) -> ToolResult:
        """Execute the options data fetch."""
        try:
            import yfinance as yf
            
            t = yf.Ticker(ticker)
            
            # Get available expirations
            expirations = t.options
            if not expirations:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No options available for {ticker}"
                )
            
            if data_type == "expirations":
                data = self._get_expirations(ticker, expirations)
            else:
                # Determine which expiration to use
                if expiration == "nearest" or expiration is None:
                    exp_date = expirations[0]
                elif expiration in expirations:
                    exp_date = expiration
                else:
                    # Find closest
                    target = datetime.strptime(expiration, "%Y-%m-%d").date()
                    exp_date = min(expirations, 
                                   key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date() - target).days))
                
                chain = t.option_chain(exp_date)
                info = t.info
                spot_price = info.get("regularMarketPrice") or info.get("previousClose")
                
                if data_type == "atm_iv":
                    data = self._get_atm_iv(ticker, exp_date, chain, spot_price, option_type)
                elif data_type == "chain_summary":
                    data = self._get_chain_summary(ticker, exp_date, chain, spot_price)
                elif data_type == "iv_skew":
                    data = self._get_iv_skew(ticker, exp_date, chain, spot_price)
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Unknown data_type: {data_type}"
                    )
            
            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"{ticker} Options - {data_type}",
                    data=data,
                    source="Yahoo Finance Options"
                ),
                metadata={"ticker": ticker, "data_type": data_type}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Options data fetch failed: {str(e)}"
            )
    
    def _get_expirations(self, ticker: str, expirations: tuple) -> dict:
        """List available expiration dates."""
        today = date.today()
        
        exp_info = []
        for exp in expirations[:20]:  # Limit to first 20
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            days_to_exp = (exp_date - today).days
            exp_info.append({
                "date": exp,
                "days_to_expiration": days_to_exp
            })
        
        return {
            "ticker": ticker,
            "num_expirations": len(expirations),
            "expirations": exp_info,
            "nearest": expirations[0] if expirations else None,
            "furthest_shown": expirations[min(19, len(expirations)-1)] if expirations else None
        }
    
    def _get_atm_iv(self, ticker: str, exp_date: str, chain, spot_price: float, option_type: str) -> dict:
        """Get ATM implied volatility."""
        result = {
            "ticker": ticker,
            "expiration": exp_date,
            "spot_price": spot_price,
            "days_to_expiration": (datetime.strptime(exp_date, "%Y-%m-%d").date() - date.today()).days
        }
        
        if option_type in ["calls", "both"]:
            calls = chain.calls.copy()
            if not calls.empty and spot_price:
                calls["dist"] = abs(calls["strike"] - spot_price)
                atm_call = calls.loc[calls["dist"].idxmin()]
                result["call_atm_strike"] = float(atm_call["strike"])
                result["call_atm_iv"] = float(atm_call["impliedVolatility"]) if not np.isnan(atm_call["impliedVolatility"]) else None
                result["call_atm_bid"] = float(atm_call["bid"]) if "bid" in atm_call else None
                result["call_atm_ask"] = float(atm_call["ask"]) if "ask" in atm_call else None
        
        if option_type in ["puts", "both"]:
            puts = chain.puts.copy()
            if not puts.empty and spot_price:
                puts["dist"] = abs(puts["strike"] - spot_price)
                atm_put = puts.loc[puts["dist"].idxmin()]
                result["put_atm_strike"] = float(atm_put["strike"])
                result["put_atm_iv"] = float(atm_put["impliedVolatility"]) if not np.isnan(atm_put["impliedVolatility"]) else None
                result["put_atm_bid"] = float(atm_put["bid"]) if "bid" in atm_put else None
                result["put_atm_ask"] = float(atm_put["ask"]) if "ask" in atm_put else None
        
        # Average IV if both available
        if result.get("call_atm_iv") and result.get("put_atm_iv"):
            result["avg_atm_iv"] = (result["call_atm_iv"] + result["put_atm_iv"]) / 2
        
        return result
    
    def _get_chain_summary(self, ticker: str, exp_date: str, chain, spot_price: float) -> dict:
        """Get summary statistics for the options chain."""
        calls = chain.calls
        puts = chain.puts
        
        def chain_stats(df, name):
            if df.empty:
                return {}
            
            iv_col = df["impliedVolatility"].dropna()
            return {
                f"{name}_count": len(df),
                f"{name}_strike_range": [float(df["strike"].min()), float(df["strike"].max())],
                f"{name}_avg_iv": float(iv_col.mean()) if len(iv_col) > 0 else None,
                f"{name}_max_iv": float(iv_col.max()) if len(iv_col) > 0 else None,
                f"{name}_min_iv": float(iv_col.min()) if len(iv_col) > 0 else None,
                f"{name}_total_volume": int(df["volume"].sum()) if "volume" in df else None,
                f"{name}_total_oi": int(df["openInterest"].sum()) if "openInterest" in df else None,
            }
        
        result = {
            "ticker": ticker,
            "expiration": exp_date,
            "spot_price": spot_price,
            "days_to_expiration": (datetime.strptime(exp_date, "%Y-%m-%d").date() - date.today()).days
        }
        result.update(chain_stats(calls, "calls"))
        result.update(chain_stats(puts, "puts"))
        
        return result
    
    def _get_iv_skew(self, ticker: str, exp_date: str, chain, spot_price: float) -> dict:
        """Get IV skew analysis (OTM puts vs OTM calls)."""
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        
        result = {
            "ticker": ticker,
            "expiration": exp_date,
            "spot_price": spot_price,
            "days_to_expiration": (datetime.strptime(exp_date, "%Y-%m-%d").date() - date.today()).days
        }
        
        if calls.empty or puts.empty or not spot_price:
            result["error"] = "Insufficient data for skew calculation"
            return result
        
        # OTM puts: strikes below spot
        otm_puts = puts[puts["strike"] < spot_price * 0.95].copy()
        # OTM calls: strikes above spot
        otm_calls = calls[calls["strike"] > spot_price * 1.05].copy()
        
        if not otm_puts.empty:
            result["otm_put_avg_iv"] = float(otm_puts["impliedVolatility"].dropna().mean())
            result["deepest_otm_put_strike"] = float(otm_puts["strike"].min())
        
        if not otm_calls.empty:
            result["otm_call_avg_iv"] = float(otm_calls["impliedVolatility"].dropna().mean())
            result["deepest_otm_call_strike"] = float(otm_calls["strike"].max())
        
        if result.get("otm_put_avg_iv") and result.get("otm_call_avg_iv"):
            result["put_call_iv_ratio"] = result["otm_put_avg_iv"] / result["otm_call_avg_iv"]
            result["skew_interpretation"] = (
                "Bearish (puts expensive)" if result["put_call_iv_ratio"] > 1.1 
                else "Bullish (calls expensive)" if result["put_call_iv_ratio"] < 0.9
                else "Neutral"
            )
        
        return result
