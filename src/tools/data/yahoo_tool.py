"""
Yahoo Finance Data Tool.

Fetches price data, historical prices, and basic info for any ticker.
Supports stocks, ETFs, futures (ES=F, GC=F), and indices (^VIX).
"""
import numpy as np
from datetime import datetime
from typing import Optional

from ..base import DataTool, ToolResult


class YahooDataTool(DataTool):
    """
    General Yahoo Finance data fetching tool.
    
    Returns research-style reports with current prices,
    historical data, and basic statistics.
    """
    
    name = "get_yahoo_data"
    description = """
Fetch price data, historical prices, or basic info for any ticker via Yahoo Finance.

Supports:
- Stocks: AAPL, NVDA, MSFT, etc.
- ETFs: SPY, HYG, QQQ, etc.
- Futures: ES=F (S&P), GC=F (Gold), CL=F (Crude), NQ=F (Nasdaq)
- Indices: ^VIX, ^GSPC (S&P 500), ^IXIC (Nasdaq)

Use this for:
- Getting current prices
- Fetching historical price data
- Getting basic security info
- Calculating returns and volatility
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol (e.g., AAPL, ES=F, ^VIX)"
            },
            "data_type": {
                "type": "string",
                "enum": ["price", "history", "info", "returns"],
                "default": "price",
                "description": "Type of data: 'price' for current, 'history' for OHLCV, 'info' for metadata, 'returns' for return stats"
            },
            "period": {
                "type": "string",
                "default": "60d",
                "description": "For history/returns: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"
            }
        },
        "required": ["ticker"]
    }
    
    async def execute(
        self,
        ticker: str,
        data_type: str = "price",
        period: str = "60d"
    ) -> ToolResult:
        """Execute the Yahoo Finance data fetch."""
        try:
            import yfinance as yf
            
            t = yf.Ticker(ticker)
            
            if data_type == "price":
                data = self._get_current_price(t, ticker)
            elif data_type == "history":
                data = self._get_history(t, ticker, period)
            elif data_type == "info":
                data = self._get_info(t, ticker)
            elif data_type == "returns":
                data = self._get_returns(t, ticker, period)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown data_type: {data_type}"
                )
            
            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"{ticker} - {data_type.title()}",
                    data=data,
                    source="Yahoo Finance"
                ),
                metadata={"ticker": ticker, "data_type": data_type}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Yahoo Finance fetch failed: {str(e)}"
            )
    
    def _get_current_price(self, t, ticker: str) -> dict:
        """Get current price and basic quote data."""
        info = t.info
        
        price = info.get("regularMarketPrice") or info.get("previousClose")
        prev_close = info.get("previousClose")
        
        change = None
        change_pct = None
        if price and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100
        
        return {
            "ticker": ticker,
            "price": price,
            "previous_close": prev_close,
            "change": change,
            "change_percent": change_pct,
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange"),
            "quote_type": info.get("quoteType"),
        }
    
    def _get_history(self, t, ticker: str, period: str) -> dict:
        """Get historical OHLCV data."""
        hist = t.history(period=period)
        
        if hist.empty:
            return {"error": "No historical data available"}
        
        # Return summary stats and recent values
        return {
            "ticker": ticker,
            "period": period,
            "data_points": len(hist),
            "date_range": {
                "start": str(hist.index[0].date()),
                "end": str(hist.index[-1].date())
            },
            "latest": {
                "date": str(hist.index[-1].date()),
                "open": float(hist["Open"].iloc[-1]),
                "high": float(hist["High"].iloc[-1]),
                "low": float(hist["Low"].iloc[-1]),
                "close": float(hist["Close"].iloc[-1]),
                "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist else None
            },
            "statistics": {
                "high_period": float(hist["High"].max()),
                "low_period": float(hist["Low"].min()),
                "avg_close": float(hist["Close"].mean()),
                "avg_volume": float(hist["Volume"].mean()) if "Volume" in hist else None
            },
            "recent_closes": [float(x) for x in hist["Close"].tail(5).tolist()]
        }
    
    def _get_info(self, t, ticker: str) -> dict:
        """Get comprehensive security info."""
        info = t.info
        
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName"),
            "quote_type": info.get("quoteType"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "50_day_avg": info.get("fiftyDayAverage"),
            "200_day_avg": info.get("twoHundredDayAverage"),
            "beta": info.get("beta"),
        }
    
    def _get_returns(self, t, ticker: str, period: str) -> dict:
        """Calculate return statistics."""
        hist = t.history(period=period)
        
        if hist.empty or len(hist) < 2:
            return {"error": "Insufficient data for return calculation"}
        
        prices = hist["Close"]
        returns = prices.pct_change().dropna()
        
        # Calculate various metrics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        return {
            "ticker": ticker,
            "period": period,
            "trading_days": len(returns),
            "total_return_pct": float(total_return),
            "daily_volatility": float(daily_vol),
            "annualized_volatility": float(annualized_vol),
            "best_day_pct": float(returns.max() * 100),
            "worst_day_pct": float(returns.min() * 100),
            "positive_days": int((returns > 0).sum()),
            "negative_days": int((returns < 0).sum()),
            "start_price": float(prices.iloc[0]),
            "end_price": float(prices.iloc[-1]),
        }
