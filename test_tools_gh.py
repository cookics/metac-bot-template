import asyncio
import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

async def run_tool_test(name, tool_instance, **kwargs):
    print(f"\n[TESTING] {name}...")
    try:
        result = await tool_instance.execute(**kwargs)
        if result.success:
            print(f"✅ {name} SUCCESS")
            return True
        else:
            print(f"❌ {name} FAILED: {result.error}")
            return False
    except Exception as e:
        print(f"💥 {name} EXCEPTION: {str(e)}")
        traceback.print_exc()
        return False

async def main():
    print("="*60)
    print("GITHUB ACTIONS TOOL VERIFICATION")
    print("="*60)
    
    from tools import get_all_tools, TOOL_REGISTRY
    
    # Check imports
    tools = get_all_tools()
    print(f"Successfully imported {len(tools)} tools.")
    
    results = {}
    
    # 1. Test offline/simulation tools (always safe)
    from tools.market.bonds_tool import BondsForecastTool
    results["BondsForecastTool"] = await run_tool_test("BondsForecastTool", BondsForecastTool(), metric="both", num_paths=100)
    
    from tools.market.vix_tool import VIXForecastTool
    results["VIXForecastTool"] = await run_tool_test("VIXForecastTool", VIXForecastTool(), trading_days=5, num_paths=100)
    
    from tools.forecast_tools import GetParametricDistributionCDF
    results["GetParametricDistributionCDF"] = await run_tool_test("GetParametricDistributionCDF", GetParametricDistributionCDF(), mean=10, std=2)

    # 2. Test API-based tools (require keys)
    if os.getenv("EXA_API_KEY"):
        from tools.search_tool import SearchTool
        results["SearchTool"] = await run_tool_test("SearchTool", SearchTool(), query="Test query", num_results=1)
    else:
        print("⏭️ Skipping SearchTool (EXA_API_KEY missing)")

    # 3. Test open data tools (Yahoo, Google Trends, FRED)
    # Note: FRED requires a key but some might be hardcoded or set in env
    from tools.data.yahoo_tool import YahooDataTool
    results["YahooDataTool"] = await run_tool_test("YahooDataTool", YahooDataTool(), ticker="AAPL", data_type="price")
    
    from tools.data.google_trends_tool import GoogleTrendsTool
    results["GoogleTrendsTool"] = await run_tool_test("GoogleTrendsTool", GoogleTrendsTool(), keywords=["test"], days_back=1)

    from tools.data.fred_tool import FREDDataTool
    results["FREDDataTool"] = await run_tool_test("FREDDataTool", FREDDataTool(), series_id="DGS10")

    # 4. Test Crypto/Prediction Markets
    from tools.market.polymarket_tool import PolyMarketSearchTool
    results["PolyMarketSearchTool"] = await run_tool_test("PolyMarketSearchTool", PolyMarketSearchTool(), search_terms=["Trump"])

    from tools.data.manifold_markets_tool import ManifoldMarketsTool
    results["ManifoldMarketsTool"] = await run_tool_test("ManifoldMarketsTool", ManifoldMarketsTool(), term="Fed")

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tools passed")
    
    if passed < total:
        print("\nSome tools failed. Check the logs above for details.")
        sys.exit(1)
    else:
        print("\nAll tested tools are functional!")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
