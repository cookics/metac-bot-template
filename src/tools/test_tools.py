"""
Test script for tool calling infrastructure.

Run from the src directory:
    python -m tools.test_tools
"""
import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_bonds_tool():
    """Test the bonds forecast tool."""
    print("\n" + "="*60)
    print("Testing BondsForecastTool")
    print("="*60)
    
    from tools.market.bonds_tool import BondsForecastTool
    
    tool = BondsForecastTool()
    print(f"Tool name: {tool.name}")
    print(f"Description: {tool.description[:100]}...")
    
    # Test execution
    result = await tool.execute(metric="both", num_paths=1000, seed=42)
    
    if result.success:
        print("\n✓ Execution successful!")
        data = result.data
        
        if "treasury_10y_yield" in data:
            yield_data = data["treasury_10y_yield"]
            print(f"\n10Y Treasury Yield:")
            print(f"  Median (p50): {yield_data['percentiles']['p50']:.2f}%")
            print(f"  5th-95th: [{yield_data['percentiles']['p5']:.2f}, {yield_data['percentiles']['p95']:.2f}]")
        
        if "hy_oas" in data:
            oas_data = data["hy_oas"]
            print(f"\nHigh Yield OAS:")
            print(f"  Median (p50): {oas_data['percentiles']['p50']:.2f}%")
            print(f"  5th-95th: [{oas_data['percentiles']['p5']:.2f}, {oas_data['percentiles']['p95']:.2f}]")
    else:
        print(f"\n✗ Execution failed: {result.error}")
    
    return result.success


async def test_spreads_tool():
    """Test the spreads forecast tool."""
    print("\n" + "="*60)
    print("Testing SpreadsForecastTool")
    print("="*60)
    
    from tools.market.spreads_tool import SpreadsForecastTool
    
    tool = SpreadsForecastTool()
    
    # Test NVDA vs AAPL
    result = await tool.execute(asset1="NVDA", asset2="AAPL", num_paths=1000, seed=42)
    
    if result.success:
        print("\n✓ NVDA vs AAPL spread:")
        print(f"  Median: {result.data['percentiles']['p50']:.2f} pp")
        print(f"  5th-95th: [{result.data['percentiles']['p5']:.2f}, {result.data['percentiles']['p95']:.2f}]")
    else:
        print(f"\n✗ Failed: {result.error}")
    
    return result.success


async def test_vix_tool():
    """Test the VIX maximum forecast tool."""
    print("\n" + "="*60)
    print("Testing VIXForecastTool")
    print("="*60)
    
    from tools.market.vix_tool import VIXForecastTool
    
    tool = VIXForecastTool()
    result = await tool.execute(trading_days=8, num_paths=1000, seed=42)
    
    if result.success:
        print("\n✓ VIX Maximum Intraday:")
        print(f"  Median: {result.data['percentiles']['p50']:.1f}")
        print(f"  5th-95th: [{result.data['percentiles']['p5']:.1f}, {result.data['percentiles']['p95']:.1f}]")
        print(f"\nTail probabilities:")
        for k, v in result.data['tail_probabilities'].items():
            print(f"  {k}: {v:.1%}")
    else:
        print(f"\n✗ Failed: {result.error}")
    
    return result.success


async def test_yahoo_tool():
    """Test the Yahoo Finance data tool."""
    print("\n" + "="*60)
    print("Testing YahooDataTool")
    print("="*60)
    
    from tools.data.yahoo_tool import YahooDataTool
    
    tool = YahooDataTool()
    
    # Test current price
    result = await tool.execute(ticker="AAPL", data_type="price")
    
    if result.success:
        print("\n✓ AAPL Price Data:")
        data = result.data["data"]
        print(f"  Price: ${data.get('price', 'N/A')}")
        print(f"  Change: {data.get('change_percent', 'N/A'):.2f}%" if data.get('change_percent') else "  Change: N/A")
    else:
        print(f"\n✗ Failed: {result.error}")
    
    return result.success


async def test_options_tool():
    """Test the Options data tool."""
    print("\n" + "="*60)
    print("Testing OptionsDataTool")
    print("="*60)
    
    from tools.data.options_tool import OptionsDataTool
    
    tool = OptionsDataTool()
    
    # Test ATM IV
    result = await tool.execute(ticker="SPY", data_type="atm_iv")
    
    if result.success:
        print("\n✓ SPY ATM IV:")
        data = result.data["data"]
        if data.get("avg_atm_iv"):
            print(f"  Average ATM IV: {data['avg_atm_iv']:.1%}")
        if data.get("call_atm_iv"):
            print(f"  Call ATM IV: {data['call_atm_iv']:.1%}")
        if data.get("put_atm_iv"):
            print(f"  Put ATM IV: {data['put_atm_iv']:.1%}")
    else:
        print(f"\n✗ Failed: {result.error}")
    
    return result.success


async def test_fred_tool():
    """Test the FRED data tool."""
    print("\n" + "="*60)
    print("Testing FREDDataTool")
    print("="*60)
    
    from tools.data.fred_tool import FREDDataTool
    
    tool = FREDDataTool()
    
    # Test 10Y Treasury
    result = await tool.execute(series_id="DGS10")
    
    if result.success:
        print("\n✓ DGS10 (10Y Treasury):")
        data = result.data["data"]
        print(f"  Current: {data['current_value']:.2f}%")
        print(f"  Date: {data['current_date']}")
        print(f"  1Y Mean: {data['statistics']['mean']:.2f}%")
    else:
        print(f"\n✗ Failed: {result.error}")
    
    return result.success


async def test_tool_schemas():
    """Test that tool schemas are valid for OpenRouter."""
    print("\n" + "="*60)
    print("Testing Tool Schemas (OpenRouter Format)")
    print("="*60)
    
    from tools.market import BondsForecastTool, SpreadsForecastTool, VIXForecastTool
    from tools.data import YahooDataTool, OptionsDataTool, FREDDataTool
    
    tools = [
        BondsForecastTool(),
        SpreadsForecastTool(),
        VIXForecastTool(),
        YahooDataTool(),
        OptionsDataTool(),
        FREDDataTool(),
    ]
    
    all_valid = True
    for tool in tools:
        schema = tool.to_openrouter_schema()
        
        # Check required fields
        has_type = schema.get("type") == "function"
        has_function = "function" in schema
        has_name = schema.get("function", {}).get("name") == tool.name
        has_desc = bool(schema.get("function", {}).get("description"))
        has_params = "parameters" in schema.get("function", {})
        
        valid = all([has_type, has_function, has_name, has_desc, has_params])
        status = "✓" if valid else "✗"
        print(f"  {status} {tool.name}")
        
        if not valid:
            all_valid = False
            print(f"    Missing: type={has_type}, function={has_function}, name={has_name}, desc={has_desc}, params={has_params}")
    
    return all_valid


async def main():
    """Run all tool tests."""
    print("\n" + "="*60)
    print("TOOL CALLING INFRASTRUCTURE TESTS")
    print("="*60)
    
    results = {}
    
    # Test schemas first
    results["schemas"] = await test_tool_schemas()
    
    # Test market tools
    results["bonds"] = await test_bonds_tool()
    results["spreads"] = await test_spreads_tool()
    results["vix"] = await test_vix_tool()
    
    # Test data tools
    results["yahoo"] = await test_yahoo_tool()
    results["options"] = await test_options_tool()
    results["fred"] = await test_fred_tool()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
