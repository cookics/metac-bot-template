"""
FRED Data Tool.

Fetches economic data series from FRED (Federal Reserve Economic Data).
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from ..base import DataTool, ToolResult

# FRED API Key
FRED_API_KEY = "0471f0cf265f624522e5bafa919aa2a3"


class FREDDataTool(DataTool):
    """
    FRED economic data fetching tool.
    
    Returns economic time series data with statistics.
    """
    
    name = "get_fred_data"
    description = """
Fetch economic data series from FRED (Federal Reserve Economic Data).

Common series IDs:
- DGS10: 10-Year Treasury Constant Maturity Rate
- DGS2: 2-Year Treasury Rate
- BAMLH0A0HYM2: ICE BofA US High Yield Index OAS
- UNRATE: Unemployment Rate
- CPIAUCSL: Consumer Price Index
- FEDFUNDS: Federal Funds Rate
- T10Y2Y: 10-Year Treasury Minus 2-Year (Yield Curve)
- VIXCLS: CBOE Volatility Index (VIX)
- SP500: S&P 500 Index

Returns current value, historical data, and basic statistics.
Useful for economic research and macro forecasting context.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "series_id": {
                "type": "string",
                "description": "FRED series ID (e.g., DGS10, UNRATE, BAMLH0A0HYM2)"
            },
            "start_date": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD), default: 1 year ago"
            },
            "include_changes": {
                "type": "boolean",
                "default": True,
                "description": "Whether to include change statistics"
            }
        },
        "required": ["series_id"]
    }
    
    # Common series for quick reference
    COMMON_SERIES = {
        "DGS10": "10-Year Treasury Constant Maturity Rate",
        "DGS2": "2-Year Treasury Constant Maturity Rate",
        "DGS30": "30-Year Treasury Constant Maturity Rate",
        "BAMLH0A0HYM2": "ICE BofA US High Yield Index OAS",
        "BAMLC0A0CM": "ICE BofA US Corporate Index OAS",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
        "FEDFUNDS": "Federal Funds Effective Rate",
        "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury (Yield Curve)",
        "T10Y3M": "10-Year Treasury Minus 3-Month Treasury",
        "VIXCLS": "CBOE Volatility Index",
        "SP500": "S&P 500 Index",
        "DTWEXBGS": "Trade Weighted U.S. Dollar Index",
        "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate",
        "GOLDAMGBD228NLBM": "Gold Fixing Price",
    }
    
    async def execute(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        include_changes: bool = True
    ) -> ToolResult:
        """Execute the FRED data fetch."""
        try:
            from fredapi import Fred
            
            fred = Fred(api_key=FRED_API_KEY)
            
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            # Fetch the series
            data = fred.get_series(series_id, observation_start=start_date)
            data = data.dropna()
            
            if data.empty:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No data found for series {series_id}"
                )
            
            # Get series info
            try:
                series_info = fred.get_series_info(series_id)
                series_title = series_info.get("title", series_id)
                units = series_info.get("units", "N/A")
                frequency = series_info.get("frequency", "N/A")
            except Exception:
                series_title = self.COMMON_SERIES.get(series_id, series_id)
                units = "N/A"
                frequency = "N/A"
            
            # Build result
            result = {
                "series_id": series_id,
                "title": series_title,
                "units": units,
                "frequency": frequency,
                "date_range": {
                    "start": str(data.index[0].date()),
                    "end": str(data.index[-1].date())
                },
                "data_points": len(data),
                "current_value": float(data.iloc[-1]),
                "current_date": str(data.index[-1].date()),
                "statistics": {
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "median": float(data.median())
                },
                "recent_values": [
                    {"date": str(data.index[-(i+1)].date()), "value": float(data.iloc[-(i+1)])}
                    for i in range(min(5, len(data)))
                ]
            }
            
            if include_changes:
                changes = data.diff().dropna()
                if len(changes) > 0:
                    result["changes"] = {
                        "last_change": float(changes.iloc[-1]),
                        "avg_daily_change": float(changes.mean()),
                        "daily_change_std": float(changes.std()),
                        "annualized_volatility": float(changes.std() * np.sqrt(252)),
                        "largest_increase": float(changes.max()),
                        "largest_decrease": float(changes.min())
                    }
                    
                    # Relative changes for percentage interpretation
                    rel_changes = data.pct_change().dropna()
                    if len(rel_changes) > 0:
                        result["relative_changes"] = {
                            "last_pct_change": float(rel_changes.iloc[-1] * 100),
                            "avg_daily_pct_change": float(rel_changes.mean() * 100),
                            "daily_pct_volatility": float(rel_changes.std() * 100),
                            "annualized_pct_volatility": float(rel_changes.std() * np.sqrt(252) * 100)
                        }
            
            return ToolResult(
                success=True,
                data=self.format_data_report(
                    title=f"FRED: {series_title}",
                    data=result,
                    source="FRED (Federal Reserve Economic Data)"
                ),
                metadata={"series_id": series_id, "data_points": len(data)}
            )
            
        except ImportError:
            return ToolResult(
                success=False,
                data=None,
                error="fredapi package not installed. Install with: pip install fredapi"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"FRED data fetch failed: {str(e)}"
            )
