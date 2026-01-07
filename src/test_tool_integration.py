"""
Integration test for tool-calling research agent.

This tests the full LLM -> tool -> LLM loop.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from research_agent import run_research_with_tools, TOOLS_AVAILABLE


async def test_market_research():
    """Test tool-calling with a market question."""
    print("\n" + "="*60)
    print("Testing Tool-Calling Research Agent")
    print("="*60)
    
    if not TOOLS_AVAILABLE:
        print("ERROR: Tools not available!")
        return False
    
    question = "What will be the 10-Year Treasury Yield ending value on January 30, 2026?"
    
    print(f"\nQuestion: {question}")
    print("\nRunning research with tools...")
    
    try:
        report, tool_calls = await run_research_with_tools(
            question=question,
            question_type="market",
            use_all_tools=True
        )
        
        print(f"\n✓ Research complete!")
        print(f"  Tool calls made: {len(tool_calls)}")
        
        for tc in tool_calls:
            print(f"    - {tc['tool_name']}: {'SUCCESS' if not tc.get('error') else 'ERROR'}")
        
        print(f"\n=== FINAL REPORT ===")
        print(report[:2000])
        if len(report) > 2000:
            print(f"... (truncated, total {len(report)} chars)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_market_research())
    sys.exit(0 if success else 1)
