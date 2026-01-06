"""
Backtest runner - orchestrates data collection, forecasting, and grading.

Usage:
    # Phase 1: Collect search data (expensive, run once)
    python backtest.py --collect --tournament 32813 --limit 50
    
    # Phase 2: Run forecasts using cached data
    python backtest.py --run --config baseline
    
    # Phase 3: Grade results
    python backtest.py --grade --run-id latest
"""
import argparse
import asyncio
import json
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

# Add root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from metaculus_api import (
    get_resolved_questions_from_tournament,
    sample_questions_evenly,
    extract_question_for_backtest,
    get_community_forecast,
    get_community_forecast_from_csv,
)
from cache import (
    save_search_cache,
    load_search_cache,
    is_cached,
    list_cached_questions,
    CACHE_DIR,
)
from news import exa_search_raw
from grading import grade_forecast, calculate_aggregate_scores, generate_report
from config import RESEARCH_MODEL, FORECAST_MODEL


RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"


def load_custom_prompts(run_name: str):
    """Load custom prompts from the run directory if they exist."""
    run_dir = RUNS_DIR / run_name
    prompts_path = run_dir / "prompts.py"
    
    if prompts_path.exists():
        print(f"[Backtest] Loading custom prompts from {prompts_path}")
        spec = importlib.util.spec_from_file_location("custom_prompts", prompts_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def get_run_dirs(run_name: str):
    """Get paths for a specific run."""
    run_dir = RUNS_DIR / run_name
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"
    return run_dir, results_dir, plots_dir


def ensure_run_dirs(run_name: str):
    """Create output directories for a specific run."""
    run_dir, results_dir, plots_dir = get_run_dirs(run_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def sample_balanced_by_type(questions: list[dict], n: int = 50) -> list[dict]:
    """
    Sample questions evenly by TYPE and TIME, excluding annulled questions.
    
    Args:
        questions: All resolved questions
        n: Total number to sample
    
    Returns:
        Balanced sample with ~equal distribution by type
    """
    # Filter out annulled questions first
    valid_questions = []
    for q in questions:
        resolution = q.get("question", {}).get("resolution")
        if resolution not in ["annulled", "ambiguous", None, ""]:
            valid_questions.append(q)
    
    print(f"[Backtest] Filtered to {len(valid_questions)} non-annulled questions")
    
    # Group by type
    by_type = defaultdict(list)
    for q in valid_questions:
        qtype = q.get("question", {}).get("type", "unknown")
        by_type[qtype].append(q)
    
    # Sort each group by publish time using published_at field
    for qtype in by_type:
        by_type[qtype].sort(key=lambda x: x.get("published_at", ""))
    
    # Calculate how many of each type (roughly equal, at least 1 from each)
    types = list(by_type.keys())
    n_types = len(types)
    per_type = max(1, n // n_types)
    
    sampled = []
    for qtype in types:
        available = by_type[qtype]
        count = min(per_type, len(available))
        
        if count > 0:
            # Even temporal sampling within type
            step = len(available) / count
            for i in range(count):
                idx = int(i * step)
                sampled.append(available[idx])
    
    # If we need more, fill from largest types
    while len(sampled) < n and any(by_type.values()):
        for qtype in sorted(types, key=lambda t: len(by_type[t]), reverse=True):
            if len(sampled) >= n:
                break
            if by_type[qtype]:
                sampled.append(by_type[qtype].pop())
    
    print(f"[Backtest] Sampled by type: {dict((t, sum(1 for q in sampled if q.get('question',{}).get('type')==t)) for t in types)}")
    
    return sampled[:n]


# ========================= PHASE 1: DATA COLLECTION =========================

async def collect_backtest_data(
    tournament_id: str | int,
    n_questions: int = 100,
    skip_cached: bool = True
) -> list[dict]:
    """
    Collect search data for backtesting.
    
    This is the expensive phase - runs searches for each question
    and caches results for reuse.
    
    Args:
        tournament_id: Tournament to fetch resolved questions from
        n_questions: Number of questions to sample
        skip_cached: Skip questions that are already cached
    
    Returns:
        List of collected question data
    """
    print(f"[Backtest] Fetching resolved questions from tournament {tournament_id}...")
    
    # Get resolved questions
    all_questions = get_resolved_questions_from_tournament(tournament_id, limit=n_questions * 3)
    
    if not all_questions:
        print("[Backtest] No resolved questions found!")
        return []
    
    # Sample evenly by type AND time
    sampled = sample_balanced_by_type(all_questions, n_questions)
    print(f"[Backtest] Sampled {len(sampled)} questions balanced by type")
    
    collected = []
    
    for i, post in enumerate(sampled):
        question = extract_question_for_backtest(post)
        question_id = question["question_id"]
        
        if skip_cached and is_cached(question_id):
            print(f"[{i+1}/{len(sampled)}] Skipping cached: {question['title'][:50]}...")
            cached = load_search_cache(question_id)
            collected.append(cached)
            continue
        
        print(f"[{i+1}/{len(sampled)}] Collecting: {question['title'][:50]}...")
        
        # Calculate search date (when the question was asked/published)
        publish_time = question.get("publish_time")
        resolved_at = question.get("resolved_at")
        
        if publish_time:
            # Use full timestamp to include news up to the moment asked
            search_cutoff = publish_time
        else:
            # Fallback to resolved_at if publish_time is missing
            search_cutoff = resolved_at if resolved_at else None
        
        # Run search with date filter
        try:
            search_results = exa_search_raw(
                query=question["title"],
                num_results=10,
                end_published_date=search_cutoff
            )
        except Exception as e:
            print(f"  Search failed: {e}")
            search_results = []
        
        # Save to cache
        save_search_cache(
            question_id=question_id,
            question_title=question["title"],
            search_query=question["title"],
            search_date=search_cutoff or "none",
            resolution_date=resolved_at or "unknown",
            search_results=search_results,
            crawled_pages=[],  # Not crawling for now
            metadata={
                "question_type": question["type"],
                "resolution": question["resolution"],
                "tournament_id": str(tournament_id),
                **question
            }
        )
        
        collected.append({
            "question_id": question_id,
            "search_results": search_results,
            "metadata": question
        })
        
        # Brief pause to avoid rate limits
        await asyncio.sleep(0.5)
    
    print(f"[Backtest] Collected data for {len(collected)} questions")
    return collected


# ========================= PHASE 2: RUN FORECASTS =========================

async def run_backtest(
    config_name: str = "default",
    forecast_model: str = None,
    research_model: str = None,
    temperature: float = None,
    limit: int = None,
    run_name: str = "backtest_1"
) -> dict:
    """
    Run forecasts on cached question data (all question types).
    
    Args:
        config_name: Name for this run configuration
        model: Model to use for forecasting
        temperature: Temperature setting
        limit: Max questions to forecast (for testing)
    
    Returns:
        Results dict with forecasts and metadata
    """
    from forecasting import (
        get_binary_gpt_prediction,
        get_numeric_gpt_prediction,
        get_multiple_choice_gpt_prediction
    )
    from research_agent import format_results_for_forecaster
    
    # Get cached questions
    cached_ids = list_cached_questions()
    
    if not cached_ids:
        print("[Backtest] No cached questions found. Run --collect first.")
        return {}
    
    if limit:
        cached_ids = cached_ids[:limit]
    
    # Load custom prompts if any
    custom_prompts = load_custom_prompts(run_name)
    
    print(f"[Backtest] Running forecasts on {len(cached_ids)} questions with config '{config_name}'")
    
    results = {
        "config_name": config_name,
        "forecast_model": forecast_model or FORECAST_MODEL,
        "research_model": research_model or RESEARCH_MODEL,
        "temperature": temperature,
        "started_at": datetime.now().isoformat(),
        "forecasts": []
    }
    
    consecutive_errors = 0  # Track for early stopping
    BATCH_SIZE = 10  # Process 10 questions at a time
    
    async def forecast_one(qid_data):
        """Forecast a single question - used for batching."""
        qid, cache, custom_prompts = qid_data
        
        if not cache:
            return None
        
        metadata = cache.get("metadata", {})
        question_type = metadata.get("type", metadata.get("question_type"))
        search_results = cache.get("search_results", [])
        question_details = {
            "title": cache.get("question_title"),
            "description": metadata.get("description", ""),
            "resolution_criteria": metadata.get("resolution_criteria", ""),
            "fine_print": metadata.get("fine_print", ""),
            "type": question_type,
            "options": metadata.get("options"),
            "scaling": metadata.get("scaling"),
            "open_upper_bound": metadata.get("open_upper_bound"),
            "open_lower_bound": metadata.get("open_lower_bound"),
            "post_id": metadata.get("post_id"),
            "publish_time": metadata.get("publish_time") or metadata.get("created_at"),
        }
        
        try:
            if question_type == "binary":
                forecast, comment = await get_binary_gpt_prediction(
                    question_details, 
                    num_runs=1,
                    research_data=search_results,
                    model=forecast_model,
                    prompt_template=getattr(custom_prompts, "BINARY_PROMPT_TEMPLATE", None) if custom_prompts else None
                )
            elif question_type == "numeric":
                forecast, comment = await get_numeric_gpt_prediction(
                    question_details, 
                    num_runs=1,
                    research_data=search_results,
                    model=forecast_model,
                    prompt_template=getattr(custom_prompts, "NUMERIC_PROMPT_TEMPLATE", None) if custom_prompts else None
                )
            elif question_type == "multiple_choice":
                forecast, comment = await get_multiple_choice_gpt_prediction(
                    question_details, 
                    num_runs=1,
                    research_data=search_results,
                    model=forecast_model,
                    prompt_template=getattr(custom_prompts, "MULTIPLE_CHOICE_PROMPT_TEMPLATE", None) if custom_prompts else None
                )
            else:
                return None
            
            return {
                "question_id": qid,
                "title": cache.get("question_title"),
                "question_type": question_type,
                "forecast": forecast,
                "resolution": metadata.get("resolution"),
                "question_details": question_details,
                "comment_preview": comment[:500] if comment else "",
                "search_results_used": len(search_results),
            }
            
        except Exception as e:
            return {
                "question_id": qid,
                "title": cache.get("question_title"),
                "question_type": question_type,
                "error": str(e),
            }
    
    # Prepare all question data
    all_qdata = []
    for qid in cached_ids:
        cache = load_search_cache(qid)
        if cache:
            all_qdata.append((qid, cache, custom_prompts))
    
    # Process in batches of BATCH_SIZE
    total_batches = (len(all_qdata) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(all_qdata))
        batch = all_qdata[start:end]
        
        print(f"\n[Backtest] Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} questions)...")
        
        # Run batch in parallel
        batch_results = await asyncio.gather(*[forecast_one(qd) for qd in batch])
        
        # Process results
        batch_errors = 0
        for result in batch_results:
            if result is None:
                continue
            results["forecasts"].append(result)
            if "error" in result:
                batch_errors += 1
                print(f"  ⚠ Error: {result.get('title', 'Unknown')[:40]}...")
            else:
                print(f"  ✓ {result.get('question_type')}: {result.get('title', '')[:40]}...")
        
        # Track consecutive errors across batches
        if batch_errors == len(batch):
            consecutive_errors += 1
            if consecutive_errors >= 2:
                print(f"\n[Backtest] ⚠️ STOPPING EARLY: {consecutive_errors} batches with all errors!")
                print(f"[Backtest] Saving partial results...")
                break
        else:
            consecutive_errors = 0
    
    results["completed_at"] = datetime.now().isoformat()
    results["n_forecasts"] = len([f for f in results["forecasts"] if "forecast" in f])
    
    # Count by type
    by_type = {}
    for f in results["forecasts"]:
        t = f.get("question_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
    results["forecasts_by_type"] = by_type
    
    # Save results
    results_dir, _ = ensure_run_dirs(run_name)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"run_{run_id}_{config_name}.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[Backtest] Saved results to {results_path}")
    print(f"[Backtest] Forecasts by type: {by_type}")
    
    return results


# ========================= PHASE 3: GRADING =========================

def grade_backtest_run(run_id: str = "latest", run_name: str = "backtest_1") -> dict:
    """
    Grade a backtest run with Peer Scores (vs community).
    
    Args:
        run_id: Run ID or "latest" for most recent
        run_name: Name of the run folder
    
    Returns:
        Grading results
    """
    results_dir, plots_dir = ensure_run_dirs(run_name)
    
    # Find run file
    if run_id == "latest":
        run_files = sorted(results_dir.glob("run_*.json"), reverse=True)
        if not run_files:
            print(f"[Backtest] No run files found in {results_dir}")
            return {}
        run_file = run_files[0]
    else:
        # Check if run_id is a full path or just a part of the filename
        if Path(run_id).exists():
            run_file = Path(run_id)
        else:
            run_file = results_dir / f"run_{run_id}.json"
            if not run_file.exists():
                # Try finding by partial match
                matches = list(results_dir.glob(f"run_{run_id}*.json"))
                if matches:
                    run_file = matches[0]
                else:
                    print(f"[Backtest] Could not find run file for {run_id} in {results_dir}")
                    return {}
    
    with open(run_file) as f:
        results = json.load(f)
    
    print(f"[Backtest] Grading run: {run_file.name}")
    
    grades = []
    
    for i, fc in enumerate(results.get("forecasts", [])):
        if "error" in fc or "forecast" not in fc:
            continue
        
        question_type = fc.get("question_type", "binary")
        resolution = fc.get("resolution")
        question_details = fc.get("question_details", {})
        post_id = question_details.get("post_id") or fc.get("post_id")
        
        # Fetch community forecast for Peer Score
        community_forecast = None
        publish_time = question_details.get("publish_time")
        
        if post_id:
            q_id = fc.get("question_id")
            
            # AUTONOMOUS COMMUNITY FETCHING: Try CSV, then API
            # No manual intervention needed - runs automatically for every question
            print(f"  [{i+1}] Fetching community for Q{q_id}...")
            
            # STEP 1: Try CSV download (most reliable source)
            cf_csv = get_community_forecast_from_csv(post_id, q_id)
            
            if "error" not in cf_csv:
                if question_type == "numeric":
                    community_forecast = cf_csv.get("forecast_values")
                elif question_type == "binary":
                    community_forecast = cf_csv.get("probability_yes")
                elif question_type == "multiple_choice":
                    community_forecast = cf_csv.get("probability_yes_per_category")
                    
                if community_forecast is not None:
                    print(f"      ✓ CSV success")
            
            # STEP 2: If CSV failed, try API
            if community_forecast is None:
                cf_api = get_community_forecast(post_id, at_time=None)  # Get latest
                
                if "error" not in cf_api:
                    if question_type == "numeric":
                        vals = cf_api.get("forecast_values")
                        if vals and len(vals) == 201 and not (all(v == 1.0 for v in vals) or all(v == 0.0 for v in vals)):
                            community_forecast = vals
                    elif question_type == "binary":
                        community_forecast = cf_api.get("probability_yes")
                    elif question_type == "multiple_choice":
                        community_forecast = cf_api.get("probability_yes_per_category")
                    
                    if community_forecast is not None:
                        print(f"      ✓ API success")
            
            if community_forecast is None:
                print(f"      ✗ No community data available")
        
        # Grade with community comparison
        grade = grade_forecast(
            forecast=fc["forecast"],
            resolution=resolution,
            question_type=question_type,
            question_details=question_details,
            community_forecast=community_forecast
        )
        
        # Skip if error or couldn't grade
        if "error" in grade:
            print(f"  [{i+1}] Could not grade: {fc.get('title', '')[:40]}... ({grade.get('error')})")
            continue
        
        grade["title"] = fc.get("title", "")
        grade["question_id"] = fc.get("question_id")
        grades.append(grade)
    
    # Calculate aggregates
    agg = calculate_aggregate_scores(grades)
    
    # Generate report
    report = generate_report(grades, results.get("config_name", "unknown"))
    
    print(report)
    
    # Save grades
    grades_path = run_file.with_suffix(".grades.json")
    with open(grades_path, "w") as f:
        json.dump({
            "run_file": str(run_file),
            "grades": grades,
            "aggregate": agg,
        }, f, indent=2)
    
    print(f"[Backtest] Saved grades to {grades_path}")
    
    # Generate detailed text tables
    try:
        from grading import generate_detailed_tables
        tables = generate_detailed_tables(grades, results)
        for filename, content in tables.items():
            table_path = plots_dir / filename
            with open(table_path, "w") as f:
                f.write(content)
            print(f"[Backtest] Saved summary table to {table_path}")
    except Exception as e:
        print(f"[Backtest] Failed to generate detailed tables: {e}")
    
    # Generate visualizations
    try:
        from visualization import generate_all_plots
        plots = generate_all_plots(grades, results.get("forecasts", []), plots_dir=plots_dir)
        if plots:
            print(f"[Backtest] Generated {len(plots)} visualization plots in {plots_dir}")
    except Exception as e:
        print(f"[Backtest] Visualization failed: {e}")
    
    return {"grades": grades, "aggregate": agg, "report": report}


# ========================= CLI =========================

async def main():
    parser = argparse.ArgumentParser(description="Backtest runner for Metaculus forecasting")
    
    parser.add_argument("--collect", action="store_true", help="Phase 1: Collect search data")
    parser.add_argument("--run", action="store_true", help="Phase 2: Run forecasts")
    parser.add_argument("--grade", action="store_true", help="Phase 3: Grade results")
    
    parser.add_argument("--tournament", type=str, default="32813", help="Tournament ID")
    parser.add_argument("--limit", type=int, default=100, help="Number of questions")
    parser.add_argument("--config", type=str, default="default", help="Config name for run")
    parser.add_argument("--run-id", type=str, default="latest", help="Run ID to grade")
    parser.add_argument("--run-name", type=str, default="backtest_1", help="Name for the run folder")
    parser.add_argument("--forecast-model", type=str, help="Override forecasting model")
    parser.add_argument("--research-model", type=str, help="Override research model")
    
    args = parser.parse_args()
    
    if args.collect:
        await collect_backtest_data(args.tournament, args.limit)
    
    elif args.run:
        await run_backtest(
            config_name=args.config, 
            limit=args.limit, 
            run_name=args.run_name,
            forecast_model=args.forecast_model,
            research_model=args.research_model
        )
    
    elif args.grade:
        grade_backtest_run(args.run_id, run_name=args.run_name)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
