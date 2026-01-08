"""
General Data Tools for Research.

These tools fetch data and return research-style reports
for the research agent to synthesize.
"""

from .yahoo_tool import YahooDataTool
from .options_tool import OptionsDataTool
from .fred_tool import FREDDataTool
from .google_trends_tool import GoogleTrendsTool

__all__ = ["YahooDataTool", "OptionsDataTool", "FREDDataTool", "GoogleTrendsTool"]
