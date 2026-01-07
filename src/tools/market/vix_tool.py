"""
VIX Maximum Intraday Forecast Tool (Q6).

Models the MAXIMUM intraday VIX reading over a time window.

Methodology:
1. VIX follows mean-reverting Ornstein-Uhlenbeck process
2. Add jump diffusion for sudden spikes (VIX can spike 50%+ in a day)
3. Simulate many paths, take MAX over each path
4. Add intraday premium (intraday highs > daily closes)

Key insight: We're modeling the PATH MAXIMUM, not terminal distribution.
"""
import numpy as np
from datetime import datetime
from typing import Tuple

from ..base import MarketForecastTool, ToolResult


class VIXForecastTool(MarketForecastTool):
    """
    Forecast tool for VIX maximum intraday questions.
    
    Uses Ornstein-Uhlenbeck + jump diffusion to simulate VIX paths,
    then returns the distribution of path maximums.
    """
    
    name = "forecast_vix_max"
    description = """
Simulate the MAXIMUM intraday VIX reading over a time window.
Uses Ornstein-Uhlenbeck + jump diffusion model with fat tails.

Key insight: Returns distribution of PATH MAXIMUMS, not terminal values.
VIX is highly volatile and mean-reverting with sudden spike potential.

Use this for questions about:
- Maximum VIX level over a period
- VIX spike probabilities
- Volatility regime questions
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "trading_days": {
                "type": "integer",
                "default": 8,
                "description": "Number of trading days to simulate"
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
        trading_days: int = 8,
        num_paths: int = 10000,
        seed: int = 42
    ) -> ToolResult:
        """Execute the VIX maximum forecast simulation."""
        try:
            rng = np.random.default_rng(seed)
            
            samples, assumptions = self._simulate_vix_max(
                num_paths, trading_days, rng
            )
            
            result = self.format_cdf_result(samples, "Q6_VIX_MAX", assumptions)
            
            # Add tail probabilities - these are important for VIX questions
            result["tail_probabilities"] = {
                "P(max >= 20)": float(np.mean(samples >= 20)),
                "P(max >= 25)": float(np.mean(samples >= 25)),
                "P(max >= 30)": float(np.mean(samples >= 30)),
                "P(max >= 35)": float(np.mean(samples >= 35)),
                "P(max >= 40)": float(np.mean(samples >= 40)),
                "P(max >= 50)": float(np.mean(samples >= 50)),
            }
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "trading_days": trading_days,
                    "num_paths": num_paths,
                    "measure": "path_maximum"
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"VIX forecast failed: {str(e)}"
            )
    
    def _get_vix_data(self) -> Tuple[float, float]:
        """Fetch current VIX and estimate futures level."""
        try:
            import yfinance as yf
            
            t = yf.Ticker("^VIX")
            info = t.info
            current = float(info.get("regularMarketPrice") or info.get("previousClose") or 17)
            
            # VIX futures typically trade at premium (contango)
            futures_est = max(current + 1.5, 18.0)
            
            return current, futures_est
        except Exception:
            return 17.0, 18.5
    
    def _simulate_vix_max(
        self, 
        n_paths: int, 
        trading_days: int, 
        rng
    ) -> Tuple[np.ndarray, dict]:
        """Simulate VIX paths using OU + Jump Diffusion, return maximum per path."""
        
        current_vix, expected_level = self._get_vix_data()
        
        # OU Parameters (calibrated to historical VIX)
        kappa = 3.0      # Mean reversion speed (annualized)
        theta = expected_level  # Long-term mean = futures level
        sigma = 0.90     # Vol of vol (90% annualized)
        
        # Jump parameters (for fat tails)
        jump_prob = 0.02    # 2% chance per day of a jump
        jump_mean = 0.35    # Average jump = +35%
        jump_std = 0.20     # Jump std
        
        # Time grid (~8 sub-steps per day for intraday granularity)
        n_steps = trading_days * 8
        dt = (trading_days / 252) / n_steps
        
        # Simulate paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = current_vix
        
        for step in range(1, n_steps + 1):
            prev = paths[:, step - 1]
            
            # OU mean reversion
            drift = kappa * (theta - prev) * dt
            diffusion = sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
            
            # Next step (OU)
            next_val = prev + drift + prev * diffusion
            
            # Jump component
            jump_occurs = rng.random(n_paths) < (jump_prob * dt * 252)
            jump_size = 1 + rng.normal(jump_mean, jump_std, n_paths)
            jump_size = np.maximum(jump_size, 0.7)  # Floor at -30%
            
            # Apply jumps
            next_val = np.where(jump_occurs, prev * jump_size, next_val)
            
            # VIX floor at 9 (historical min ~9.14)
            paths[:, step] = np.maximum(next_val, 9.0)
        
        # Get PATH MAXIMUM (not terminal)
        path_max = np.max(paths, axis=1)
        
        # Intraday premium: actual intraday highs exceed our discrete simulation
        # Add 5-15% premium
        intraday_factor = 1.05 + 0.10 * rng.random(n_paths)
        max_values = path_max * intraday_factor
        
        assumptions = {
            "current_vix": f"{current_vix:.1f}",
            "expected_level": f"{expected_level:.1f} (futures contango)",
            "mean_reversion": f"κ = {kappa:.1f}",
            "vol_of_vol": f"σ = {sigma:.0%}",
            "jump_prob": f"{jump_prob:.0%}/day",
            "model": "OU + Jump Diffusion",
            "measure": "PATH MAXIMUM (not terminal)"
        }
        
        return max_values, assumptions
