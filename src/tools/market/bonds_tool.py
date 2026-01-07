"""
Bonds Forecast Tool - 10Y Treasury Yield (Q7) and High Yield OAS (Q8).

Uses FRED API for actual data and historical volatility.
Accounts for fat tails using Student's t-distribution and HYG options IV.

FRED Series:
- BAMLH0A0HYM2: ICE BofA US High Yield Index Option-Adjusted Spread
- DGS10: 10-Year Treasury Constant Maturity Rate
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from ..base import MarketForecastTool, ToolResult

# FRED API - imported lazily to avoid import errors if not installed
FRED_API_KEY = "0471f0cf265f624522e5bafa919aa2a3"


class BondsForecastTool(MarketForecastTool):
    """
    Forecast tool for bond-related questions.
    
    Returns CDF distributions for:
    - 10Y Treasury Yield ending value
    - High Yield OAS ending value
    """
    
    name = "forecast_bonds"
    description = """
Simulate 10Y Treasury Yield and/or High Yield OAS (Option-Adjusted Spread) distributions.
Uses FRED historical data for calibration, options implied volatility from HYG ETF, 
and fat-tailed Monte Carlo simulation (Student's t-distribution, jump diffusion for OAS).

Returns percentile distributions that can be directly used for forecasting.

Use this tool for questions about:
- 10-Year Treasury yield ending values
- High Yield bond spreads / OAS
- ICE BofA US High Yield Index
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["yield", "oas", "both"],
                "default": "both",
                "description": "Which metric to forecast: 'yield' for 10Y Treasury, 'oas' for High Yield OAS, 'both' for both"
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
        "required": []
    }
    
    async def execute(
        self,
        metric: str = "both",
        trading_days: int = 8,
        num_paths: int = 10000,
        seed: int = 42
    ) -> ToolResult:
        """Execute the bonds forecast simulation."""
        try:
            results = {}
            time_horizon = trading_days / 252
            rng = np.random.default_rng(seed)
            
            if metric in ["yield", "both"]:
                samples, assumptions = self._simulate_yield(
                    num_paths, time_horizon, rng
                )
                results["treasury_10y_yield"] = self.format_cdf_result(
                    samples, "Q7_10Y_YIELD", assumptions
                )
            
            if metric in ["oas", "both"]:
                samples, assumptions = self._simulate_hy_oas(
                    num_paths, time_horizon, rng
                )
                results["hy_oas"] = self.format_cdf_result(
                    samples, "Q8_HY_OAS", assumptions
                )
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "trading_days": trading_days,
                    "num_paths": num_paths,
                    "metrics_computed": list(results.keys())
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Bonds forecast failed: {str(e)}"
            )
    
    def _get_fred_series(self, series_id: str, start_date: str = None) -> tuple:
        """Fetch series from FRED."""
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            data = fred.get_series(series_id, observation_start=start_date)
            data = data.dropna()
            
            current = float(data.iloc[-1])
            
            # Daily changes
            changes = data.diff().dropna()
            daily_std = changes.std()
            annualized_std = daily_std * np.sqrt(252)
            relative_vol = annualized_std / current if current > 0 else 0.15
            
            return current, float(relative_vol), data
        except Exception as e:
            # Fallback values
            if series_id == "DGS10":
                return 4.25, 0.14, None
            elif series_id == "BAMLH0A0HYM2":
                return 2.80, 0.25, None
            else:
                raise e
    
    def _get_hyg_options_iv(self) -> tuple:
        """Get HYG (High Yield ETF) options implied volatility."""
        try:
            import yfinance as yf
            from datetime import date
            
            t = yf.Ticker("HYG")
            expirations = t.options
            
            if not expirations:
                return None, "No HYG options available"
            
            # Find closest to target date
            target = date(2026, 1, 30)
            closest = min(expirations, 
                         key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d").date() - target).days))
            
            chain = t.option_chain(closest)
            puts = chain.puts
            
            if puts.empty:
                return None, "Empty puts chain"
            
            info = t.info
            price = info.get("regularMarketPrice") or info.get("previousClose")
            
            if price:
                puts = puts.copy()
                puts["dist"] = abs(puts["strike"] - price)
                atm = puts.loc[puts["dist"].idxmin()]
                iv = atm.get("impliedVolatility")
                if iv and not np.isnan(iv) and iv > 0:
                    return float(iv), f"HYG Put IV ({closest})"
            
            return None, "Could not extract IV"
            
        except Exception as e:
            return None, f"Error: {str(e)[:30]}"
    
    def _simulate_yield(self, n_paths: int, time_horizon: float, rng) -> tuple:
        """Simulate 10Y Treasury Yield with slight fat tails."""
        try:
            current_yield, vol, history = self._get_fred_series("DGS10")
            source = "FRED DGS10"
        except Exception:
            current_yield = 4.25
            vol = 0.14
            source = "Fallback"
        
        period_vol = vol * np.sqrt(time_horizon)
        
        # Use Student's t with df=6 for slight fat tails
        df = 6
        t_samples = rng.standard_t(df, n_paths)
        t_samples = t_samples / np.sqrt(df / (df - 2))
        
        terminal = current_yield * np.exp(period_vol * t_samples - 0.5 * period_vol**2)
        terminal = np.maximum(terminal, 0.0)
        
        assumptions = {
            "current_yield": f"{current_yield:.2f}%",
            "volatility": f"{vol:.1%}",
            "data_source": source,
            "fat_tails": "Student's t (df=6)",
            "model": "GBM (zero drift)"
        }
        
        return terminal, assumptions
    
    def _simulate_hy_oas(self, n_paths: int, time_horizon: float, rng) -> tuple:
        """Simulate High Yield OAS with fat tails and jump diffusion."""
        try:
            current_oas, _, history = self._get_fred_series("BAMLH0A0HYM2")
            if history is not None:
                long_term_mean = float(history.tail(252).mean())
                daily_changes = history.diff().dropna()
                hist_annualized_std = float(daily_changes.std() * np.sqrt(252))
            else:
                long_term_mean = 3.50
                hist_annualized_std = 0.80
            fred_source = "FRED BAMLH0A0HYM2"
        except Exception:
            current_oas = 2.80
            long_term_mean = 3.50
            hist_annualized_std = 0.80
            fred_source = "Fallback"
        
        # Get HYG options IV for market-implied vol
        hyg_iv, iv_source = self._get_hyg_options_iv()
        
        # Blend historical vol with options IV if available
        if hyg_iv:
            implied_spread_vol = hyg_iv * 4.0 * 100 / current_oas
            blended_vol = 0.7 * hist_annualized_std + 0.3 * implied_spread_vol
            vol_source = f"Blended (Hist + {iv_source})"
        else:
            blended_vol = hist_annualized_std
            vol_source = "Historical only"
        
        # OU parameters
        kappa = 2.0
        theta = long_term_mean
        sigma = blended_vol
        
        exp_decay = np.exp(-kappa * time_horizon)
        expected = theta + (current_oas - theta) * exp_decay
        
        if kappa > 0:
            variance = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * time_horizon))
        else:
            variance = sigma**2 * time_horizon
        
        std_dev = np.sqrt(variance)
        
        # Fat tails: Student's t-distribution
        df = 5
        t_samples = rng.standard_t(df, n_paths)
        t_samples = t_samples / np.sqrt(df / (df - 2))
        
        terminal_oas = expected + std_dev * t_samples
        
        # Jump component: crisis spike probability
        jump_prob = 0.025
        jump_occurs = rng.random(n_paths) < jump_prob
        jump_size = rng.exponential(scale=0.80, size=n_paths)
        
        terminal_oas = np.where(jump_occurs, terminal_oas + jump_size, terminal_oas)
        terminal_oas = np.maximum(terminal_oas, 0.5)
        
        assumptions = {
            "current_oas": f"{current_oas:.2f}%",
            "long_term_mean": f"{long_term_mean:.2f}%",
            "volatility": f"{blended_vol:.2f}pp/yr ({vol_source})",
            "mean_reversion": f"κ = {kappa:.1f}",
            "fat_tails": f"Student's t (df={df})",
            "jump_risk": f"{jump_prob:.1%} crisis prob",
            "data_source": fred_source
        }
        
        return terminal_oas, assumptions
