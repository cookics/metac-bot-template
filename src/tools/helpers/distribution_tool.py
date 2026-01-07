"""
General Distribution Tool.

Lets the model easily generate probability distributions (Gaussian, etc.)
without writing Python code. Returns CDF percentiles in the same format
as the market forecast tools.

This is useful for questions where the model has a belief about the
mean and uncertainty but doesn't need a full Monte Carlo simulation.
"""
import numpy as np
from datetime import datetime
from typing import Optional, Literal

from ..base import MarketForecastTool, ToolResult


class DistributionGeneratorTool(MarketForecastTool):
    """
    General tool for generating probability distributions.
    
    The model can specify a distribution type (Gaussian, log-normal, uniform, etc.)
    and its parameters, and get back a CDF in the standard format.
    """
    
    name = "generate_distribution"
    description = """
Generate a probability distribution and return CDF percentiles.

Use this when you have a belief about a quantity but don't need a full simulation.
Just specify the distribution type and parameters.

Supported distributions:
- gaussian/normal: Specify mean and std
- lognormal: Specify median and multiplicative_std (e.g., 1.5 means 50% spread)
- uniform: Specify min and max
- triangular: Specify min, mode, and max
- mixture: Two Gaussians with weights (for bimodal)

Returns percentiles in the same format as market forecast tools.
"""
    
    parameters = {
        "type": "object",
        "properties": {
            "distribution": {
                "type": "string",
                "enum": ["gaussian", "normal", "lognormal", "uniform", "triangular", "mixture"],
                "description": "Type of distribution to generate"
            },
            "mean": {
                "type": "number",
                "description": "Mean for Gaussian, or center for other distributions"
            },
            "std": {
                "type": "number",
                "description": "Standard deviation for Gaussian"
            },
            "median": {
                "type": "number",
                "description": "Median for log-normal distribution"
            },
            "multiplicative_std": {
                "type": "number",
                "description": "Multiplicative spread for log-normal (e.g., 1.5 means 50% spread)"
            },
            "min_val": {
                "type": "number",
                "description": "Minimum value for uniform/triangular"
            },
            "max_val": {
                "type": "number",
                "description": "Maximum value for uniform/triangular"
            },
            "mode": {
                "type": "number",
                "description": "Mode (peak) for triangular distribution"
            },
            "mean2": {
                "type": "number",
                "description": "Mean of second Gaussian for mixture"
            },
            "std2": {
                "type": "number",
                "description": "Std of second Gaussian for mixture"
            },
            "weight1": {
                "type": "number",
                "description": "Weight of first Gaussian in mixture (0-1), default 0.5"
            },
            "question_context": {
                "type": "string",
                "description": "Brief description of what this distribution represents"
            },
            "num_samples": {
                "type": "integer",
                "default": 10000,
                "description": "Number of samples to generate"
            }
        },
        "required": ["distribution"]
    }
    
    async def execute(
        self,
        distribution: str,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        median: Optional[float] = None,
        multiplicative_std: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        mode: Optional[float] = None,
        mean2: Optional[float] = None,
        std2: Optional[float] = None,
        weight1: float = 0.5,
        question_context: str = "",
        num_samples: int = 10000,
        **kwargs
    ) -> ToolResult:
        """Generate the specified distribution."""
        try:
            rng = np.random.default_rng()
            
            if distribution in ["gaussian", "normal"]:
                if mean is None or std is None:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Gaussian requires 'mean' and 'std' parameters"
                    )
                samples = rng.normal(mean, std, num_samples)
                assumptions = {
                    "distribution": "Gaussian (Normal)",
                    "mean": mean,
                    "std": std
                }
                
            elif distribution == "lognormal":
                if median is None:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Log-normal requires 'median' parameter"
                    )
                mult_std = multiplicative_std or 1.3
                # Convert to log-normal parameters
                mu = np.log(median)
                sigma = np.log(mult_std)
                samples = rng.lognormal(mu, sigma, num_samples)
                assumptions = {
                    "distribution": "Log-Normal",
                    "median": median,
                    "multiplicative_std": mult_std
                }
                
            elif distribution == "uniform":
                if min_val is None or max_val is None:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Uniform requires 'min_val' and 'max_val' parameters"
                    )
                samples = rng.uniform(min_val, max_val, num_samples)
                assumptions = {
                    "distribution": "Uniform",
                    "min": min_val,
                    "max": max_val
                }
                
            elif distribution == "triangular":
                if min_val is None or max_val is None or mode is None:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Triangular requires 'min_val', 'mode', and 'max_val' parameters"
                    )
                samples = rng.triangular(min_val, mode, max_val, num_samples)
                assumptions = {
                    "distribution": "Triangular",
                    "min": min_val,
                    "mode": mode,
                    "max": max_val
                }
                
            elif distribution == "mixture":
                if mean is None or std is None or mean2 is None or std2 is None:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="Mixture requires 'mean', 'std', 'mean2', and 'std2' parameters"
                    )
                # Generate mixture of two Gaussians
                n1 = int(num_samples * weight1)
                n2 = num_samples - n1
                samples1 = rng.normal(mean, std, n1)
                samples2 = rng.normal(mean2, std2, n2)
                samples = np.concatenate([samples1, samples2])
                rng.shuffle(samples)
                assumptions = {
                    "distribution": "Mixture of Two Gaussians",
                    "component1": f"N({mean}, {std}) weight={weight1:.2f}",
                    "component2": f"N({mean2}, {std2}) weight={1-weight1:.2f}"
                }
                
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown distribution: {distribution}"
                )
            
            # Add context if provided
            if question_context:
                assumptions["context"] = question_context
            
            # Format as CDF result
            result = self.format_cdf_result(
                samples=samples,
                question_id=f"generated_{distribution}",
                assumptions=assumptions
            )
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "distribution": distribution,
                    "num_samples": num_samples
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Distribution generation failed: {str(e)}"
            )
