"""
Verification Demo - Shows tool calling in action with logged output.

This script runs three example questions that exercise different tools,
logs the model's thoughts and tool interactions, and saves to a file.
"""
import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import get_all_tools
from tools.executor import run_tool_calling_loop
from config import RESEARCH_MODEL, RESEARCH_TEMP

# Output directory for logs
LOG_DIR = Path(__file__).parent.parent / "logs" / "tool_demos"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# System prompt for research with tools
DEMO_SYSTEM_PROMPT = """You are a Research Agent with access to specialized tools.

Your job is to gather data for forecasting. You have these tools:
1. forecast_bonds, forecast_spread, forecast_vix_max - Monte Carlo simulations
2. get_yahoo_data, get_options_data, get_fred_data - Financial data
3. generate_distribution - Create custom probability distributions
4. search_web - Search for news and context

For market questions, use forecast tools to get probability distributions.
For general questions, combine data tools and search.

After gathering data, write a SHORT REPORT synthesizing your findings.
Be concise - the forecaster needs key facts, not essays."""


# Example questions that exercise different tool combinations
DEMO_QUESTIONS = [
    {
        "id": "Q1_treasury_yield",
        "question": "What will the 10-Year Treasury Yield be at the end of January 2026?",
        "expected_tools": ["forecast_bonds", "get_fred_data"],
        "description": "Market question - should use bonds forecast + FRED data"
    },
    {
        "id": "Q2_nvda_vs_aapl",
        "question": "How much will NVDA outperform or underperform AAPL by the end of January 2026?",
        "expected_tools": ["forecast_spread", "get_yahoo_data"],
        "description": "Spread question - should use spreads forecast + Yahoo data"
    },
    {
        "id": "Q3_vix_spike",
        "question": "What is the probability that VIX exceeds 30 at some point in January 2026?",
        "expected_tools": ["forecast_vix_max", "get_options_data"],
        "description": "VIX question - should use VIX forecast + options data"
    },
]


class DemoLogger:
    """Logs tool calling interactions to file."""
    
    def __init__(self, question_id: str):
        self.question_id = question_id
        self.log_path = LOG_DIR / f"demo_{question_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.entries = []
    
    def log(self, entry_type: str, data: dict):
        """Add a log entry."""
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,
            "data": data
        })
    
    def save(self):
        """Save logs to file."""
        with open(self.log_path, "w") as f:
            json.dump({
                "question_id": self.question_id,
                "log_path": str(self.log_path),
                "entries": self.entries
            }, f, indent=2, default=str)
        return self.log_path


async def run_demo_question(question_info: dict) -> dict:
    """Run a single demo question and log all interactions."""
    logger = DemoLogger(question_info["id"])
    
    print(f"\n{'='*60}")
    print(f"DEMO: {question_info['id']}")
    print(f"{'='*60}")
    print(f"Question: {question_info['question']}")
    print(f"Expected tools: {question_info['expected_tools']}")
    print()
    
    # Log question
    logger.log("question", {
        "id": question_info["id"],
        "question": question_info["question"],
        "expected_tools": question_info["expected_tools"]
    })
    
    # Get all tools
    tools = get_all_tools()
    logger.log("available_tools", {"tools": [t.name for t in tools]})
    
    print(f"[Demo] Running with {len(tools)} tools...")
    
    # Run the tool calling loop
    try:
        final_response, tool_calls = await run_tool_calling_loop(
            initial_prompt=question_info["question"],
            tools=tools,
            model=RESEARCH_MODEL,
            temperature=RESEARCH_TEMP,
            max_iterations=5,
            system_prompt=DEMO_SYSTEM_PROMPT
        )
        
        # Log tool calls
        for i, tc in enumerate(tool_calls):
            logger.log("tool_call", {
                "index": i,
                "tool_name": tc["tool_name"],
                "arguments": tc["arguments"],
                "success": tc.get("error") is None,
                "error": tc.get("error"),
                "result_type": type(tc.get("result")).__name__ if tc.get("result") else None
            })
            print(f"  Tool call {i+1}: {tc['tool_name']}")
        
        # Log final response
        logger.log("final_response", {
            "response": final_response,
            "length": len(final_response)
        })
        
        print(f"\n[Demo] Complete! Made {len(tool_calls)} tool calls")
        print(f"\n--- FINAL RESPONSE ---")
        print(final_response[:1500])
        if len(final_response) > 1500:
            print(f"... (truncated, {len(final_response)} chars total)")
        
        # Check if expected tools were used
        tools_used = [tc["tool_name"] for tc in tool_calls]
        expected = set(question_info["expected_tools"])
        used = set(tools_used)
        
        logger.log("verification", {
            "tools_used": tools_used,
            "expected_tools": list(expected),
            "all_expected_used": expected.issubset(used),
            "unexpected_tools": list(used - expected)
        })
        
        if expected.issubset(used):
            print(f"\n✓ Used expected tools: {expected}")
        else:
            print(f"\n⚠ Missing tools: {expected - used}")
            print(f"  Used instead: {used}")
        
        # Save log
        log_path = logger.save()
        print(f"\n📄 Log saved: {log_path}")
        
        return {
            "success": True,
            "question_id": question_info["id"],
            "tools_used": tools_used,
            "log_path": str(log_path)
        }
        
    except Exception as e:
        logger.log("error", {"error": str(e)})
        logger.save()
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "question_id": question_info["id"],
            "error": str(e)
        }


async def main():
    """Run all demo questions."""
    print("\n" + "="*60)
    print("TOOL CALLING VERIFICATION DEMO")
    print("="*60)
    print(f"\nModel: {RESEARCH_MODEL}")
    print(f"Log directory: {LOG_DIR}")
    print(f"\nRunning {len(DEMO_QUESTIONS)} demo questions...\n")
    
    results = []
    for q in DEMO_QUESTIONS:
        result = await run_demo_question(q)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['question_id']}: {r.get('tools_used', r.get('error', 'N/A'))}")
    
    print(f"\nLogs saved to: {LOG_DIR}")
    
    return all(r["success"] for r in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
