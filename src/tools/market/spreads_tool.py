"""
Spreads Forecast Tool - Return spread simulations (Q1-Q5).

Q1: NVDA vs AAPL
Q2: NVDA vs MSFT
Q3: Gold vs S&P 500
Q4: Crude vs S&P 500
Q5: Nasdaq vs S&P 500

Methodology: Correlated GBM using Options IV (stocks) or Historical Vol (futures)
"""
import numpy as np
from datetime import datetime, date
from typing import Optional, Tuple

from ..base import MarketForecastTool, ToolResult


class SpreadsForecastTool(MarketForecastTool):
    """
    Forecast tool for return spread questions.
    
    Simulates the return spread between two assets using correlated GBM.
    Returns CDF distribution of the spread in percentage points.
    """
    
    name = "forecast_spread"
    description = """
Simulate the return spread between two assets using correlated Geometric Brownian Motion.
Uses implied volatility from options (for stocks) or historical volatility (for futures).

Supports common spread questions:
- NVDA vs AAPL (Q1)
- NVDA vs MSFT (Q2)  
- Gold (GC=F) vs S&P 500 (ES=F) (Q3)
- Crude Oil (CL=F) vs S&P 500 (ES=F) (Q4)
- Nasdaq (NQ=F) vs S&P 500 (ES=F) (Q5)

Returns percentile distribution of the spread (asset1_return - asset2_return) in percentage points.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "asset1": {
                "type": "string",
                "description": "First asset ticker (e.g., NVDA, GC=F for Gold futures)"
            },
            "asset2": {
                "type": "string",
                "description": "Second asset ticker (e.g., AAPL, ES=F for S&P futures)"
            },
            "trading_days": {
                "type": "integer",
                "default": 8,
                "description": "Number of trading days to simulate forward"
            },
            "num_paths": {
                "type": "integer",
                "default": 10000,
                "description": "Number of Monte Carlo simulation paths"
            },
            "seed": {
                "type": "integer",
                "default": 42,
                "description": "Random seed for reproducibility"
            }
        },
        "required": ["asset1", "asset2"]
    }
    
    # Pre-defined spread questions for quick lookup
    PRESET_SPREADS = {
        "Q1": ("NVDA", "AAPL", "NVDA vs AAPL"),
        "Q2": ("NVDA", "MSFT", "NVDA vs MSFT"),
        "Q3": ("GC=F", "ES=F", "Gold vs S&P 500"),
        "Q4": ("CL=F", "ES=F", "Crude vs S&P 500"),
        "Q5": ("NQ=F", "ES=F", "Nasdaq vs S&P 500"),
    }
    
    async def execute(
        self,
        asset1: str,
        asset2: str,
        trading_days: int = 8,
        num_paths: int = 10000,
        seed: int = 42
    ) -> ToolResult:
        """Execute the spread forecast simulation."""
        try:
            time_horizon = trading_days / 252
            rng = np.random.default_rng(seed)
            
            samples, assumptions = self._simulate_spread(
                asset1, asset2, num_paths, time_horizon, rng
            )
            
            # Identify question ID if this is a preset
            question_id = None
            for qid, (a1, a2, _) in self.PRESET_SPREADS.items():
                if a1 == asset1 and a2 == asset2:
                    question_id = qid
                    break
            
            if question_id is None:
                question_id = f"{asset1}_vs_{asset2}"
            
            result = self.format_cdf_result(samples, question_id, assumptions)
            result["spread_description"] = f"{asset1} return minus {asset2} return (percentage points)"
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "asset1": asset1,
                    "asset2": asset2,
                    "trading_days": trading_days,
                    "num_paths": num_paths
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Spread forecast failed: {str(e)}"
            )
    
    def _get_price(self, ticker: str) -> float:
        """Get current price for a ticker."""
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        return float(info.get("regularMarketPrice") or info.get("previousClose") or 0)
    
    def _get_volatility(self, ticker: str) -> Tuple[float, str]:
        """Get IV from options if available, else historical vol."""
        import yfinance as yf
        
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            
            if expirations:
                target = date(2026, 1, 30)
                closest = min(expirations, 
                             key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date() - target).days))
                chain = t.option_chain(closest)
                calls = chain.calls
                
                if not calls.empty:
                    info = t.info
                    price = info.get("regularMarketPrice") or info.get("previousClose")
                    
                    if price:
                        calls = calls.copy()
                        calls["dist"] = abs(calls["strike"] - price)
                        atm = calls.loc[calls["dist"].idxmin()]
                        iv = atm.get("impliedVolatility")
                        
                        # Sanity check: IV should be at least 1% (0.01)
                        # Yahoo sometimes returns placeholder values like 1e-05
                        if iv and not np.isnan(iv) and iv > 0.01:
                            return float(iv), f"Options IV ({closest})"
        except Exception:
            pass
        
        # Fallback to historical volatility
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="70d")["Close"]
            returns = hist.pct_change().dropna().tail(60)
            hv = float(returns.std() * np.sqrt(252))
            return hv, "Historical Vol"
        except Exception:
            return 0.30, "Fallback Vol"
    
    def _get_correlation(self, t1: str, t2: str, days: int = 60) -> float:
        """Get correlation between two tickers."""
        import yfinance as yf
        
        try:
            h1 = yf.Ticker(t1).history(period="70d")["Close"].pct_change().dropna()
            h2 = yf.Ticker(t2).history(period="70d")["Close"].pct_change().dropna()
            common = h1.index.intersection(h2.index)
            
            if len(common) < 20:
                return 0.5
            
            return float(np.corrcoef(h1.loc[common].tail(days), h2.loc[common].tail(days))[0, 1])
        except Exception:
            return 0.5
    
    def _simulate_spread(
        self, 
        t1: str, 
        t2: str, 
        n_paths: int, 
        time_horizon: float, 
        rng
    ) -> Tuple[np.ndarray, dict]:
        """Simulate spread using correlated GBM."""
        
        p1 = self._get_price(t1)
        p2 = self._get_price(t2)
        v1, v1_src = self._get_volatility(t1)
        v2, v2_src = self._get_volatility(t2)
        corr = self._get_correlation(t1, t2)
        
        # Period volatility
        pv1 = v1 * np.sqrt(time_horizon)
        pv2 = v2 * np.sqrt(time_horizon)
        
        # Correlated normals
        Z1 = rng.standard_normal(n_paths)
        Z2 = corr * Z1 + np.sqrt(1 - corr**2) * rng.standard_normal(n_paths)
        
        # Log-normal returns
        r1 = np.exp(pv1 * Z1 - 0.5 * pv1**2) - 1
        r2 = np.exp(pv2 * Z2 - 0.5 * pv2**2) - 1
        
        # Spread in percentage points
        spread = (r1 - r2) * 100
        
        assumptions = {
            t1: f"${p1:.2f}, vol={v1:.1%} ({v1_src})",
            t2: f"${p2:.2f}, vol={v2:.1%} ({v2_src})",
            "correlation": f"{corr:.2f} (60-day hist)",
            "method": "Correlated GBM"
        }
        
        return spread, assumptions
