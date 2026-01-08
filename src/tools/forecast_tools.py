import numpy as np
from scipy.stats import skewnorm, norm
from typing import Any
from .base import BaseTool, ToolResult

class GetParametricDistributionCDF(BaseTool):
    """
    Generates a smooth CDF for a Parametric Distribution (Normal or SkewNormal).
    
    Useful for creating a baseline forecast distribution when you have an estimate 
    of the Mean and Standard Deviation (and optionally Skew).
    """
    
    name = "get_parametric_cdf"
    description = (
        "Returns a smooth CDF (percentiles) for a distribution with the given Mean, "
        "Standard Deviation, and Skew. Use this to generate a baseline forecast."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mean": {
                "type": "number",
                "description": "The expected value (mean) of the distribution."
            },
            "std": {
                "type": "number",
                "description": "The standard deviation (uncertainty/width) of the distribution."
            },
            "skew": {
                "type": "number",
                "description": "Shape parameter (alpha). 0=Normal. >0 for Right Skew (long right tail). <0 for Left Skew. Range -10 to 10.",
                "default": 0.0
            },
            "lower_bound": {
                "type": "number",
                "description": "Optional lower bound. Percentiles below this will be clipped (mass handled by CDF)."
            },
            "upper_bound": {
                "type": "number",
                "description": "Optional upper bound."
            }
        },
        "required": ["mean", "std"]
    }

    async def execute(
        self, 
        mean: float, 
        std: float, 
        skew: float = 0.0,
        lower_bound: float = None,
        upper_bound: float = None
    ) -> dict[str, Any]:
        try:
            # 1. Calculate SkewNormal parameters (loc, scale) to match target Mean/Std
            # Given alpha (skew input), we need to solve for loc and scale.
            
            # alpha is the shape parameter
            alpha = skew
            
            # Correlation coefficient delta
            delta = alpha / np.sqrt(1 + alpha**2)
            
            # Standard deviation adjustment factor
            # Var = scale^2 * (1 - 2*delta^2/pi)
            # Scale = Std / sqrt(1 - 2*delta^2/pi)
            factor = 1 - (2 * delta**2 / np.pi)
            if factor <= 0:
                # Should not happen for real alpha
                factor = 1.0
            
            scale = std / np.sqrt(factor)
            
            # Mean adjustment factor
            # Mean = loc + scale * delta * sqrt(2/pi)
            # Loc = Mean - scale * delta * sqrt(2/pi)
            mean_adjustment = scale * delta * np.sqrt(2 / np.pi)
            loc = mean - mean_adjustment
            
            # 2. Generate percentiles
            # We want p1, p5, p10...p99
            percentiles = {}
            probs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
            
            for p in probs:
                q = p / 100.0
                if alpha == 0:
                    val = norm.ppf(q, loc=mean, scale=std)
                else:
                    val = skewnorm.ppf(q, alpha, loc=loc, scale=scale)
                
                # Clip if bounds provided
                if lower_bound is not None:
                    val = max(val, lower_bound)
                if upper_bound is not None:
                    val = min(val, upper_bound)
                    
                percentiles[f"p{p}"] = float(val)
            
            data = {
                "percentiles": percentiles,
                "parameters": {
                    "input_mean": mean,
                    "input_std": std,
                    "input_skew": skew,
                    "fitted_loc": float(loc),
                    "fitted_scale": float(scale),
                    "fitted_alpha": float(alpha)
                },
                "explanation": (
                    f"Generated {('Normal' if skew==0 else 'SkewNormal')} distribution "
                    f"with Mean={mean}, Std={std}. "
                    f"{'Right skewed (tail > mean)' if skew > 0 else 'Left skewed (tail < mean)' if skew < 0 else 'Symmetric'}."
                )
            }
            return ToolResult(success=True, data=data)
            
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
