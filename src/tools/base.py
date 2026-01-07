"""
Base classes for tool definitions.

Provides OpenRouter-compatible tool schemas and standard result types.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolResult:
    """Standard result from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_message_content(self) -> str:
        """Format result for inclusion in tool message."""
        if not self.success:
            return f"Tool execution failed: {self.error}"
        
        if isinstance(self.data, dict):
            import json
            return json.dumps(self.data, indent=2, default=str)
        return str(self.data)


class BaseTool(ABC):
    """
    Base class for all agent tools.
    
    Subclasses must define:
    - name: Tool identifier
    - description: What the tool does (shown to the model)
    - parameters: JSON Schema for tool parameters
    - execute(): Async method to run the tool
    """
    
    name: str
    description: str
    parameters: dict  # JSON Schema
    
    def to_openrouter_schema(self) -> dict:
        """
        Return OpenRouter-compatible tool definition.
        
        Format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "What the tool does",
                "parameters": { JSON Schema }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Returns:
            ToolResult with success status and data/error.
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"


class MarketForecastTool(BaseTool):
    """
    Base class for market forecast tools that return CDF distributions.
    
    These tools run Monte Carlo simulations and return percentile data
    that can be directly used by the forecaster model.
    """
    
    def format_cdf_result(
        self, 
        samples: "np.ndarray",
        question_id: str,
        assumptions: dict
    ) -> dict:
        """
        Format simulation samples into CDF percentiles for the forecaster.
        
        Returns a dict with:
        - percentiles: {1, 5, 10, 25, 50, 75, 90, 95, 99}
        - statistics: {mean, std, min, max}
        - assumptions: model assumptions for transparency
        """
        import numpy as np
        
        percentile_keys = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
                          55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]
        percentiles = {f"p{p}": float(np.percentile(samples, p)) for p in percentile_keys}
        
        return {
            "question_id": question_id,
            "type": "forecast_distribution",
            "percentiles": percentiles,
            "statistics": {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "min": float(np.min(samples)),
                "max": float(np.max(samples)),
                "n_samples": len(samples)
            },
            "assumptions": assumptions,
            "generated_at": datetime.now().isoformat()
        }


class DataTool(BaseTool):
    """
    Base class for general data tools that return research-style reports.
    
    These tools fetch data and format it as readable summaries
    for the research agent to synthesize.
    """
    
    def format_data_report(
        self,
        title: str,
        data: dict,
        source: str
    ) -> dict:
        """
        Format data into a research-style report.
        
        Returns a dict with:
        - title: What data was fetched
        - source: Data source (Yahoo, FRED, etc.)
        - data: The actual data
        - summary: Human-readable summary
        """
        return {
            "type": "data_report",
            "title": title,
            "source": source,
            "data": data,
            "fetched_at": datetime.now().isoformat()
        }
