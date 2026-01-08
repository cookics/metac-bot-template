"""
Tool Registry and Exports for Agentic Tool Calling.

This module provides:
- Base tool classes with OpenRouter-compatible schemas
- Registry of all available tools
- Helper functions to get tools for specific question types
"""

from .base import BaseTool, ToolResult
from .executor import run_tool_calling_loop, call_llm_with_tools

# Import tool classes
from .market import BondsForecastTool, SpreadsForecastTool, VIXForecastTool
from .data import YahooDataTool, OptionsDataTool, FREDDataTool
from .forecast_tools import GetParametricDistributionCDF
from .helpers import DistributionGeneratorTool
from .search_tool import SearchTool
from .crawl_tool import CrawlTool

# Registry of all available tools
TOOL_REGISTRY = {
    # Market forecast tools (return CDF distributions)
    "forecast_bonds": BondsForecastTool,
    "forecast_spread": SpreadsForecastTool,
    "forecast_vix_max": VIXForecastTool,
    # General data tools (return research reports)
    "get_yahoo_data": YahooDataTool,
    "get_options_data": OptionsDataTool,
    "get_fred_data": FREDDataTool,
    # Helper tools
    "generate_distribution": DistributionGeneratorTool,
    # Search and crawl tools
    "search_web": SearchTool,
    "crawl_urls": CrawlTool,
    # Forecast distribution tool (parametric)
    "get_parametric_cdf": GetParametricDistributionCDF,
}


def get_tool(name: str) -> BaseTool:
    """Get a tool instance by name."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOL_REGISTRY.keys())}")
    return TOOL_REGISTRY[name]()


def get_all_tools() -> list[BaseTool]:
    """Get instances of all available tools."""
    return [cls() for cls in TOOL_REGISTRY.values()]


def get_market_tools() -> list[BaseTool]:
    """Get market forecast tools (bonds, spreads, VIX)."""
    market_names = ["forecast_bonds", "forecast_spread", "forecast_vix_max"]
    return [TOOL_REGISTRY[name]() for name in market_names]


def get_data_tools() -> list[BaseTool]:
    """Get general data tools (Yahoo, options, FRED)."""
    data_names = ["get_yahoo_data", "get_options_data", "get_fred_data"]
    return [TOOL_REGISTRY[name]() for name in data_names]


def get_research_tools() -> list[BaseTool]:
    """Get tools for general research (search, crawl, data)."""
    names = ["search_web", "crawl_urls", "get_yahoo_data", "get_options_data", "get_fred_data"]
    return [TOOL_REGISTRY[name]() for name in names]


def get_helper_tools() -> list[BaseTool]:
    """Get helper tools (distribution generator)."""
    helper_names = ["generate_distribution"]
    return [TOOL_REGISTRY[name]() for name in helper_names]


def get_tool_schemas() -> list[dict]:
    """Get OpenRouter-compatible schemas for all tools."""
    return [tool.to_openrouter_schema() for tool in get_all_tools()]
