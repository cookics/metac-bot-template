"""
Single Question Test Utility - Runs forecasting for one specific question and logs everything.
Supports Metaculus IDs or custom question strings.

Usage:
  python src/test_single_question.py --id 3479 --runs 1
  python src/test_single_question.py --id 3479 --runs 1 --submit
  python src/test_single_question.py --question "Will Bitcoin hit $100k by February 2026?"
"""
import asyncio
import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# Add project root and src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.config import RESEARCH_MODEL, FORECAST_MODEL, USE_TOOLS
from src.metaculus_api import (
    get_post_details, 
    post_question_prediction,
    create_forecast_payload
)
from src.research_agent import run_research_pipeline
from src.forecasting import (
    detect_question_type
)
from src.llm import call_llm
from src.prompts import (
    BINARY_PROMPT_TEMPLATE,
    NUMERIC_PROMPT_TEMPLATE,
    MULTIPLE_CHOICE_PROMPT_TEMPLATE
)

# Output directory for logs
LOG_DIR = ROOT_DIR / "logs" / "question_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class QuestionTestLogger:
    """Logs the entire forecasting process for a single question."""
    
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = LOG_DIR
        
        self.data = {
            "identifier": identifier,
            "start_time": self.start_time.isoformat(),
            "config": {
                "research_model": RESEARCH_MODEL,
                "forecast_model": FORECAST_MODEL,
                "use_tools": USE_TOOLS
            },
            "question": {},
            "research": {},
            "research_messages": [],
            "grok_calls": [],       # Log all Grok (research) I/O
            "opus_calls": [],       # Log all Opus (forecast) I/O
            "forecasting": {},
            "final_result": {},
            "costs": {              # Cost tracking
                "research_tokens": 0,
                "research_cost": 0.0,
                "forecast_tokens": 0,
                "forecast_cost": 0.0,
                "exa_cost": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "details": []
            },
            "tool_usage": {},
            "submitted_to_metaculus": False
        }
    
    def log_question(self, question_data: dict):
        self.data["question"] = question_data
        
    def log_research(self, research_data: dict):
        self.data["research"] = research_data
        if "messages" in research_data:
            self.data["research_messages"] = research_data["messages"]
            # Extract Grok I/O from messages
            self._extract_grok_calls(research_data["messages"])
        
        # Capture Exa cost and tool usage
        if "exa_cost" in research_data:
            self.data["costs"]["exa_cost"] = research_data["exa_cost"]
        if "tool_usage" in research_data:
            self.data["tool_usage"] = research_data["tool_usage"]
    
    def _extract_grok_calls(self, messages: list):
        """Extract input/output pairs from research messages."""
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                # This is an input to Grok
                call = {"input": msg.get("content", "")[:5000]}  # Truncate for log size
                # Look for the assistant response
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    resp = messages[i + 1]
                    call["output"] = resp.get("content", "")[:5000]
                    if resp.get("tool_calls"):
                        call["tool_calls"] = [
                            {"name": tc.get("function", {}).get("name"), 
                             "args": tc.get("function", {}).get("arguments", "")[:500]}
                            for tc in resp.get("tool_calls", [])
                        ]
                self.data["grok_calls"].append(call)
    
    def log_opus_call(self, input_prompt: str, output_response: str):
        """Log a single Opus (forecasting) call."""
        self.data["opus_calls"].append({
            "input": input_prompt,
            "output": output_response
        })
        
    def log_forecasting(self, forecast_result: dict):
        self.data["forecasting"] = forecast_result
    
    def log_submission(self, success: bool, response: str = None):
        self.data["submitted_to_metaculus"] = success
        self.data["submission_response"] = response
    
    def log_cost(self, phase: str, tokens: int, cost: float, details: dict = None):
        """Log cost information for a phase (research or forecast)."""
        if phase == "research":
            self.data["costs"]["research_tokens"] += tokens
            self.data["costs"]["research_cost"] += cost
        elif phase == "forecast":
            self.data["costs"]["forecast_tokens"] += tokens
            self.data["costs"]["forecast_cost"] += cost
        
        self.data["costs"]["total_tokens"] += tokens
        self.data["costs"]["total_cost"] += cost
        
        if details:
            details["phase"] = phase
            self.data["costs"]["details"].append(details)
        
    def save(self) -> str:
        """Saves JSON log and raw text log."""
        self.data["end_time"] = datetime.now().isoformat()
        self.data["duration_seconds"] = (datetime.now() - self.start_time).total_seconds()
        
        log_path = os.path.join(self.log_dir, f"test_{self.identifier}_{self.timestamp}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)
            
        # Save detailed raw text log
        raw_path = os.path.join(self.log_dir, f"test_{self.identifier}_{self.timestamp}_raw.txt")
        with open(raw_path, "w", encoding='utf-8') as f:
            self._write_raw_log(f)
        print(f"[Logger] Raw log saved: {raw_path}")
            
        return log_path
    
    def _write_raw_log(self, f):
        """Write human-readable raw log."""
        f.write(f"{'='*80}\n")
        f.write(f"RAW I/O LOG: {self.data['question'].get('title', 'Unknown')}\n")
        f.write(f"Timestamp: {self.timestamp}\n")
        f.write(f"Research Model: {RESEARCH_MODEL}\n")
        f.write(f"Forecast Model: {FORECAST_MODEL}\n")
        f.write(f"{'='*80}\n\n")
        
        # Section 1: Grok (Research) Calls
        f.write(f"{'#'*80}\n")
        f.write(f"# GROK (RESEARCH MODEL) CALLS\n")
        f.write(f"{'#'*80}\n\n")
        
        for i, call in enumerate(self.data.get("grok_calls", []), 1):
            f.write(f"--- GROK CALL {i} ---\n\n")
            f.write(f"[INPUT TO GROK]\n")
            f.write(f"{'-'*40}\n")
            f.write(f"{call.get('input', 'N/A')}\n")
            f.write(f"{'-'*40}\n\n")
            
            f.write(f"[OUTPUT FROM GROK]\n")
            f.write(f"{'-'*40}\n")
            f.write(f"{call.get('output', 'N/A')}\n")
            
            if call.get("tool_calls"):
                f.write(f"\n[TOOL CALLS MADE]\n")
                for tc in call["tool_calls"]:
                    f.write(f"  - {tc['name']}({tc['args'][:200]}...)\n")
            f.write(f"{'-'*40}\n\n")
        
        # Section 2: Research Message History (full)
        f.write(f"\n{'#'*80}\n")
        f.write(f"# FULL RESEARCH MESSAGE HISTORY\n")
        f.write(f"{'#'*80}\n\n")
        
        for msg in self.data.get("research_messages", []):
            role = msg.get("role", "unknown").upper()
            f.write(f"[{role}]\n")
            
            if msg.get("content"):
                f.write(f"{msg['content']}\n")
            
            if msg.get("tool_calls"):
                f.write("\n[TOOL CALLS]\n")
                for tc in msg["tool_calls"]:
                    f.write(f"  - {tc.get('function', {}).get('name')}({tc.get('function', {}).get('arguments')[:200]}...)\n")
            
            f.write(f"\n{'-'*60}\n\n")
        
        # Section 3: Opus (Forecasting) Calls
        f.write(f"\n{'#'*80}\n")
        f.write(f"# OPUS (FORECASTING MODEL) CALLS\n")
        f.write(f"{'#'*80}\n\n")
        
        for i, call in enumerate(self.data.get("opus_calls", []), 1):
            f.write(f"--- OPUS CALL {i} ---\n\n")
            f.write(f"[INPUT TO OPUS (PROMPT)]\n")
            f.write(f"{'-'*40}\n")
            f.write(f"{call.get('input', 'N/A')}\n")
            f.write(f"{'-'*40}\n\n")
            
            f.write(f"[OUTPUT FROM OPUS (RESPONSE)]\n")
            f.write(f"{'-'*40}\n")
            f.write(f"{call.get('output', 'N/A')}\n")
            f.write(f"{'-'*40}\n\n")
        
        # Section 4: Final Result
        f.write(f"\n{'#'*80}\n")
        f.write(f"# FINAL RESULT\n")
        f.write(f"{'#'*80}\n\n")
        f.write(f"Type: {self.data.get('final_result', {}).get('type', 'N/A')}\n")
        f.write(f"Success: {self.data.get('final_result', {}).get('success', 'N/A')}\n")
        f.write(f"Submitted: {self.data.get('submitted_to_metaculus', False)}\n")
        
        forecast = self.data.get('final_result', {}).get('forecast', 'N/A')
        if isinstance(forecast, list):
            f.write(f"Forecast (CDF): [{forecast[0]:.4f}, ..., {forecast[-1]:.4f}] ({len(forecast)} points)\n")
        else:
            f.write(f"Forecast: {forecast}\n")
        
        # Section 5: Cost Summary
        f.write(f"\n{'#'*80}\n")
        f.write(f"# COST SUMMARY\n")
        f.write(f"{'#'*80}\n\n")
        costs = self.data.get("costs", {})
        f.write(f"Research Phase:\n")
        f.write(f"  - Tokens: {costs.get('research_tokens', 0):,}\n")
        f.write(f"  - Cost: ${costs.get('research_cost', 0):.6f}\n")
        f.write(f"\nForecast Phase:\n")
        f.write(f"  - Tokens: {costs.get('forecast_tokens', 0):,}\n")
        f.write(f"  - Cost: ${costs.get('forecast_cost', 0):.6f}\n")
        f.write(f"\nExa API:\n")
        f.write(f"  - Cost: ${costs.get('exa_cost', 0):.6f}\n")
        f.write(f"\nTOTAL:\n")
        f.write(f"  - Total Tokens: {costs.get('total_tokens', 0):,}\n")
        total_with_exa = costs.get('total_cost', 0) + costs.get('exa_cost', 0)
        f.write(f"  - Total Cost: ${total_with_exa:.6f}\n")
        
        # Section 6: Tool Usage
        tool_usage = self.data.get("tool_usage", {})
        if tool_usage:
            f.write(f"\n{'#'*80}\n")
            f.write(f"# TOOL USAGE\n")
            f.write(f"{'#'*80}\n\n")
            for tool_name, count in sorted(tool_usage.items()):
                f.write(f"  {tool_name}: {count} call(s)\n")


async def run_forecast_with_logging(
    question_details: dict,
    research_result: dict,
    logger: QuestionTestLogger,
    num_runs: int = 1
):
    """Run forecasting with full I/O logging for Opus."""
    import datetime as dt
    import re
    import numpy as np
    from src.forecasting import (
        extract_probability_from_response_as_percentage_not_decimal,
        extract_percentiles_from_response,
        extract_date_percentiles_from_response,
        extract_option_probabilities_from_response,
        generate_continuous_cdf,
        generate_multiple_choice_forecast
    )
    from src.config import FORECAST_TEMP, FORECAST_THINKING, USE_TOOLS
    from src.prompts import FORECAST_SYSTEM_PROMPT
    from src.tools import get_tool, run_tool_calling_loop
    
    today = dt.datetime.now().strftime("%Y-%m-%d")
    title = question_details.get("title", "")
    q_type = question_details.get("type", "binary")
    
    # Get research summary
    summary_report = research_result.get("formatted_for_forecaster", "No research available")
    
    # Select template and build format args
    format_args = {
        "title": title,
        "today": today,
        "background": question_details.get("description", ""),
        "resolution_criteria": question_details.get("resolution_criteria", ""),
        "fine_print": question_details.get("fine_print", ""),
        "summary_report": summary_report,
    }
    
    if q_type == "binary":
        template = BINARY_PROMPT_TEMPLATE
    elif q_type in ["numeric", "date"]:
        template = NUMERIC_PROMPT_TEMPLATE
        # Add bound messages for numeric/date
        scaling = question_details.get("scaling", {})
        open_upper = question_details.get("open_upper_bound", True)
        open_lower = question_details.get("open_lower_bound", True)
        upper_bound = scaling.get("range_max", 0)
        lower_bound = scaling.get("range_min", 0)
        
        format_args["upper_bound_message"] = "" if open_upper else f"The outcome can not be higher than {upper_bound}."
        format_args["lower_bound_message"] = "" if open_lower else f"The outcome can not be lower than {lower_bound}."
    else:
        template = MULTIPLE_CHOICE_PROMPT_TEMPLATE
        format_args["options"] = question_details.get("options", [])
    
    # Build prompt
    content = template.format(**format_args)

    
    results = []
    
    for run_idx in range(num_runs):
        # Call LLM (with tools if enabled for numeric questions)
        if USE_TOOLS and q_type in ["numeric", "date"]:
            print(f"[Forecast] Run {run_idx + 1}/{num_runs} (Using Parametric Tool Loop)...")
            # Initialize tools
            forecast_tools = [get_tool("get_parametric_cdf")]
            
            # Use the tool loop
            response, tool_calls, _ = await run_tool_calling_loop(
                initial_prompt=content,
                tools=forecast_tools,
                model=FORECAST_MODEL,
                temperature=FORECAST_TEMP,
                max_iterations=3,
                system_prompt=FORECAST_SYSTEM_PROMPT,
                thinking=FORECAST_THINKING
            )
            # For simplicity in this test logger, we don't return stats for tool loop yet
            # but we record that it happened
            stats = {"tool_calls": len(tool_calls)}
            result = (response, stats)
        else:
            print(f"[Forecast] Run {run_idx + 1}/{num_runs}...")
            result = await call_llm(
                content, 
                model=FORECAST_MODEL, 
                temperature=FORECAST_TEMP, 
                thinking=FORECAST_THINKING,
                return_stats=True
            )
        
        # Handle both return types
        if isinstance(result, tuple):
            response, stats = result
            # Log cost info
            tokens = stats.get("native_tokens_prompt", 0) + stats.get("native_tokens_completion", 0)
            cost = stats.get("total_cost", 0.0)
            logger.log_cost("forecast", tokens, cost, stats)
            print(f"[Forecast] Cost: ${cost:.6f} ({tokens:,} tokens)")
        else:
            response = result
        
        # Log the I/O
        logger.log_opus_call(content, response)
        
        results.append(response)
    
    # Process results based on question type
    if q_type == "binary":
        probabilities = []
        comments = []
        for i, resp in enumerate(results):
            prob = extract_probability_from_response_as_percentage_not_decimal(resp)
            probabilities.append(prob)
            comments.append(f"## Rationale {i+1}\nExtracted Probability: {prob}%\n\nGPT's Answer: {resp}\n\n")
        
        median_prob = float(np.median(probabilities)) / 100
        final_comment = f"Median Probability: {median_prob}\n\n" + "\n".join(comments)
        return median_prob, final_comment
    
    elif q_type in ["numeric", "date"]:
        cdfs = []
        comments = []
        scaling = question_details.get("scaling", {})
        
        for i, resp in enumerate(results):
            percentiles = None
            
            # --- NEW FALLBACK LOGIC: Extract from tool results if available ---
            # Try to find tool calls in the research/forecast context
            # We need to know if this run used tools
            if USE_TOOLS and "[Using Parametric Tool Loop]" in str(logger.data): # Rough check
                # Actually, we can check the stats or the response itself
                pass

            # Use date-specific extractor for date questions (converts to timestamps)
            if q_type == "date":
                percentiles = extract_date_percentiles_from_response(resp)
            else:
                percentiles = extract_percentiles_from_response(resp)
            
            # If extraction failed, look for tool results in the last loop
            # Note: We don't have easy access to tool_calls here because of the loop structure
            # Let's fix the loop to return them if needed, or just let the model handle it
            # For now, let's just make the prompt even more explicit.
            
            if not percentiles:
                 # Try one last thing: if the response itself looks like the tool returned JSON
                 import json
                 try:
                     # Remove markdown if present
                     clean_resp = resp.replace("```json", "").replace("```", "").strip()
                     data = json.loads(clean_resp)
                     if "percentiles" in data:
                         percentiles = data["percentiles"]
                         print(f"[Forecast] Extracted percentiles from JSON response/tool output.")
                 except:
                     pass

            if not percentiles:
                print(f"[Forecast] ERROR: Could not extract percentiles from Run {i+1}")
                continue

            cdf = generate_continuous_cdf(
                percentiles,
                q_type,
                question_details.get("open_upper_bound", True),
                question_details.get("open_lower_bound", True),
                scaling.get("range_max", 1.0),
                scaling.get("range_min", 0.0),
                scaling.get("zero_point"),
            )
            cdfs.append(cdf)
            comments.append(f"## Rationale {i+1}\nExtracted Percentile_values: {percentiles}\n\nGPT's Answer: {resp}\n\n")
        
        # Average CDFs
        median_cdf = [float(np.median([cdfs[j][i] for j in range(len(cdfs))])) for i in range(len(cdfs[0]))]
        final_comment = f"Median CDF: `{str(median_cdf[:5])}...`\n\n" + "\n".join(comments)
        return median_cdf, final_comment
    
    elif q_type == "multiple_choice":
        options = question_details.get("options", [])
        all_probs = []
        comments = []
        
        for i, resp in enumerate(results):
            probs = extract_option_probabilities_from_response(resp, options)
            all_probs.append(probs)
            comments.append(f"## Rationale {i+1}\nExtracted Probabilities: {probs}\n\nGPT's Answer: {resp}\n\n")
        
        # Average probabilities
        avg_probs = {}
        for opt in options:
            avg_probs[opt] = float(np.mean([p.get(opt, 0) for p in all_probs]))
        
        # Normalize
        total = sum(avg_probs.values())
        if total > 0:
            avg_probs = {k: v/total for k, v in avg_probs.items()}
        
        final_comment = f"Average Probabilities: {avg_probs}\n\n" + "\n".join(comments)
        return avg_probs, final_comment
    
    return None, "Unknown question type"


async def run_test(
    post_id: int = None, 
    custom_question: str = None, 
    num_runs: int = 1,
    submit: bool = False
):
    """Run a forecast test for a single question."""
    
    question_id = None
    
    # 1. Setup identifier and fetch question details
    if post_id:
        print(f"\n[Test] Fetching Metaculus post {post_id}...")
        post_details = get_post_details(post_id)
        question_details = post_details["question"]
        question_id = question_details.get("id")
        title = question_details["title"]
        identifier = f"metaculus_{post_id}"
    else:
        print(f"\n[Test] Using custom question: {custom_question}")
        title = custom_question
        question_details = {
            "title": custom_question,
            "type": "binary",
            "description": "Custom test question",
            "resolution_criteria": "N/A",
            "fine_print": "N/A"
        }
        identifier = "custom_" + "".join(c if c.isalnum() else "_" for c in custom_question[:20])

    logger = QuestionTestLogger(identifier)
    logger.log_question(question_details)
    
    print(f"--- Question: {title} ---")
    print(f"Type: {question_details['type']}")
    
    # 2. Run Research Pipeline (logs Grok I/O via messages)
    print(f"\n[Test] Running research pipeline...")
    q_type = detect_question_type(title)
    research_result = await run_research_pipeline(title, question_type=q_type)
    logger.log_research(research_result)
    
    print(f"Research complete. Made {len(research_result.get('tool_calls', []))} tool calls.")
    
    # 3. Run Forecasting with detailed Opus logging
    print(f"\n[Test] Running forecasting (runs={num_runs})...")
    q_type_metaculus = question_details["type"]
    
    forecast = None
    comment = None
    
    try:
        forecast, comment = await run_forecast_with_logging(
            question_details,
            research_result,
            logger,
            num_runs
        )
        
        logger.log_forecasting({
            "forecast": forecast,
            "comment": comment
        })
        
        logger.data["final_result"] = {
            "type": q_type_metaculus,
            "forecast": forecast,
            "success": True
        }
        
        print(f"\n--- Forecast: {forecast if not isinstance(forecast, list) else f'CDF ({len(forecast)} points)'} ---")
        
        # 4. Submit to Metaculus if requested
        if submit and post_id and question_id:
            print(f"\n[Test] Submitting forecast to Metaculus...")
            try:
                payload = create_forecast_payload(forecast, q_type_metaculus)
                post_question_prediction(question_id, payload)
                
                # Also post the rationale as a comment
                from src.metaculus_api import post_question_comment
                post_question_comment(post_id, comment)
                
                logger.log_submission(True, "Success")
                print(f"[Test] [SUCCESS] Forecast and comment submitted successfully to question {question_id}")
            except Exception as e:
                logger.log_submission(False, str(e))
                print(f"[Test] [FAILED] Failed to submit: {e}")
        elif submit and not post_id:
            print("[Test] Cannot submit custom questions to Metaculus")
            
    except Exception as e:
        print(f"\n[Test] Error during forecasting: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.log_forecasting({"error": str(e)})
        logger.data["final_result"] = {"success": False, "error": str(e)}

    # 5. Save and finish
    log_path = logger.save()
    print(f"\n{'='*60}")
    print(f"TEST COMPLETE")
    print(f"Log saved: {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test forecasting for a single question")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=int, help="Metaculus Post ID")
    group.add_argument("--question", type=str, help="Custom question string")
    parser.add_argument("--runs", type=int, default=1, help="Number of forecasting runs")
    parser.add_argument("--submit", action="store_true", help="Submit forecast to Metaculus")
    
    args = parser.parse_args()
    
    asyncio.run(run_test(
        post_id=args.id, 
        custom_question=args.question, 
        num_runs=args.runs,
        submit=args.submit
    ))
