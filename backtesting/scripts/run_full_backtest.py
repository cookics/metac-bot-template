#!/usr/bin/env python
"""
Full Backtest Pipeline - Runs all steps automatically.

Usage:
    python run_full_backtest.py --run-name backtest_4 --limit 50

This script chains:
1. backtest.py --run (forecasting)
2. backtest.py --grade (grading)
3. fetch_fixed_community.py (CSV community fetch)
4. gen_tables.py (table generation)
"""

import subprocess
import sys
import argparse
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable  # Use the same Python that's running this script


def run_step(step_name: str, cmd: list[str]) -> bool:
    """Run a step and return True if successful."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR.parent.parent)
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {step_name} (exit code {result.returncode})")
        return False
    
    print(f"\n✅ COMPLETED: {step_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full backtest pipeline")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the backtest run (e.g., backtest_4)")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions to sample")
    parser.add_argument("--skip-run", action="store_true", help="Skip forecasting step (reuse existing)")
    parser.add_argument("--skip-grade", action="store_true", help="Skip grading step (reuse existing)")
    parser.add_argument("--forecast-model", type=str, help="Model to use for forecasting (comma-separated for multi)")
    parser.add_argument("--research-model", type=str, help="Model to use for research")
    parser.add_argument("--config", type=str, help="Custom slug for the config (replaces model name in filenames)")
    parser.add_argument("--compare-runs", type=str, help="Comma-separated list of other run names to include in comparison")
    args = parser.parse_args()
    
    run_name = args.run_name
    limit = args.limit
    custom_config = args.config
    
    print(f"\n{'#'*60}")
    print(f"# FULL BACKTEST PIPELINE: {run_name}")
    print(f"# Limit: {limit} questions")
    print(f"{'#'*60}")
    
    # Create directories
    runs_dir = SCRIPTS_DIR.parent / "data" / "runs" / run_name
    (runs_dir / "results").mkdir(parents=True, exist_ok=True)
    (runs_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    # Step 1 & 2: Forecasting and Grading (Loop per model)
    forecast_models = args.forecast_model.split(",") if args.forecast_model else [None]
    failed = False
    
    for model_name in forecast_models:
        model_name = model_name.strip() if model_name else None
        # Create a simplified config name for filenames
        model_slug = custom_config or "default"
        if not custom_config and model_name:
            model_slug = model_name.split("/")[-1].replace(":", "_").replace(".", "_")
            
        print(f"\n>>> PROCESSING MODEL: {model_name or 'Default'} (Slug: {model_slug})")
        
        # Forecasting
        if not args.skip_run:
            cmd = [PYTHON, str(SCRIPTS_DIR / "backtest.py"), "--run", "--run-name", run_name, "--config", model_slug, "--limit", str(limit)]
            if model_name:
                cmd.extend(["--forecast-model", model_name])
            if args.research_model:
                cmd.extend(["--research-model", args.research_model])
            
            if not run_step(f"Forecasting: {model_slug}", cmd):
                failed = True
                break
        
        # Grading
        if not args.skip_grade:
            cmd = [PYTHON, str(SCRIPTS_DIR / "backtest.py"), "--grade", "--run-name", run_name, "--config", model_slug]
            if model_name:
                cmd.extend(["--forecast-model", model_name])
            if args.research_model:
                cmd.extend(["--research-model", args.research_model])
                
            if not run_step(f"Grading: {model_slug}", cmd):
                failed = True
                break

    if not failed:
        # Step 3: Fetch community forecasts via CSV (Generic for the run)
        if not run_step("Fetch Community (fetch_fixed_community.py)", 
                       [PYTHON, str(SCRIPTS_DIR / "fetch_fixed_community.py"), "--run-name", run_name]):
            failed = True
            
    if not failed:
        # Step 4: Generate tables (now handles all models in run)
        cmd = [PYTHON, str(SCRIPTS_DIR / "gen_tables.py"), "--run-name", run_name]
        if args.compare_runs:
            cmd.extend(["--compare-runs", args.compare_runs])
            
        if not run_step("Generate Tables (gen_tables.py)", cmd):
            failed = True
    
    # Summary
    print(f"\n{'#'*60}")
    if failed:
        print(f"# ❌ PIPELINE FAILED")
    else:
        print(f"# ✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"#")
        print(f"# Results: backtesting/data/runs/{run_name}/plots/")
        print(f"# Comparison: comparison_summary.txt")
    print(f"{'#'*60}\n")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
