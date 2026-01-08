"""
Test script for PolyMarketSearchTool.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.market.polymarket_tool import PolyMarketSearchTool

async def test_polymarket_search():
    print("\n" + "="*60)
    print("Testing PolyMarketSearchTool")
    print("="*60)
    
    tool = PolyMarketSearchTool()
    print(f"Tool name: {tool.name}")
    
    # Test 1: Trump search
    print("\nTest 1: Searching for 'Trump'...")
    result = await tool.execute(search_terms=["Trump"])
    
    if result.success:
        print("✓ Success!")
        markets = result.data.get("data", {}).get("markets", [])
        print(f"Found {len(markets)} markets.")
        for m in markets[:5]:
            print(f"- {m['question']} (Vol: ${m['volume']:,.0f})")
            print(f"  Probs: {m['probabilities']}")
    else:
        print(f"✗ Failed: {result.error}")

    # Test 2: Bitcoin search
    print("\nTest 2: Searching for ['Bitcoin', 'ETF']...")
    result = await tool.execute(search_terms=["Bitcoin", "ETF"])
    
    if result.success:
        print("✓ Success!")
        markets = result.data.get("data", {}).get("markets", [])
        print(f"Found {len(markets)} markets.")
        if markets:
            print(f"Top market: {markets[0]['question']}")
    else:
        print(f"✗ Failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_polymarket_search())
