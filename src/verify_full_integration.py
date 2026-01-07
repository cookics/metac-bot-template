"""
Final end-to-end verification of the integrated tool-calling pipeline.
This tests:
1. detect_question_type keywords
2. run_research_pipeline integration in forecasting functions
3. Structured data passing to the forecaster
4. Expensive mode (Grok/Claude 4.5)
"""
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from forecasting import get_numeric_gpt_prediction, detect_question_type
from config import RESEARCH_MODEL, FORECAST_MODEL, USE_TOOLS, EXPENSIVE_MODE

async def verify_pipeline():
    print(f"--- Pipeline Configuration ---")
    print(f"USE_TOOLS: {USE_TOOLS}")
    print(f"EXPENSIVE_MODE: {EXPENSIVE_MODE}")
    print(f"Research Model: {RESEARCH_MODEL}")
    print(f"Forecast Model: {FORECAST_MODEL}")
    print(f"------------------------------\n")

    # Test Market Question Detection
    test_titles = [
        "What will the 10Y Treasury Yield be on Jan 30, 2026?",
        "Will NVDA vs AAPL spread exceed 5%?",
        "Will the US unemployment rate increase by March 2026?",
        "Will SpaceX land a starship on Mars in 2026?"
    ]
    
    print("--- Testing Question Type Detection ---")
    for title in test_titles:
        q_type = detect_question_type(title)
        print(f"Title: {title[:50]}... -> Type: {q_type}")
    print()

    # Run one full forecast (Numeric Market)
    question = {
        "title": "What will be the 10-Year Treasury Yield ending value on January 30, 2026?",
        "description": "This question asks for the 10Y yield as reported by FRED (DGS10).",
        "resolution_criteria": "Resolves to the DGS10 value on Jan 30, 2026.",
        "fine_print": "None.",
        "type": "numeric",
        "scaling": {"range_min": 0, "range_max": 10, "zero_point": None},
        "open_upper_bound": True,
        "open_lower_bound": False
    }

    print("--- Running Full Market Forecast Pipeline ---")
    print("(This involves tool calling with Grok and forecasting with Claude 4.5)")
    
    # We only run 1 num_runs for verification speed
    try:
        cdf, rationale = await get_numeric_gpt_prediction(question, num_runs=1)
        
        print("\n=== VERIFICATION SUCCESS ===")
        print(f"Median of CDF: {cdf[100]:.4f}")
        print(f"Rationale Length: {len(rationale)} chars")
        print(f"Rationale Snippet: {rationale[:300]}...")
        
        # Check if rationale contains tool data headers
        if "=== 10Y TREASURY YIELD FORECAST ===" in rationale:
            print("✓ Found structured market tool data in rationale.")
        if "SYNTHESIS:" in rationale:
            print("✓ Found research synthesis in rationale.")
            
    except Exception as e:
        print(f"\n!!! VERIFICATION FAILED !!!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_pipeline())
