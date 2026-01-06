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
    args = parser.parse_args()
    
    run_name = args.run_name
    limit = args.limit
    
    print(f"\n{'#'*60}")
    print(f"# FULL BACKTEST PIPELINE: {run_name}")
    print(f"# Limit: {limit} questions")
    print(f"{'#'*60}")
    
    # Create directories
    runs_dir = SCRIPTS_DIR.parent / "data" / "runs" / run_name
    (runs_dir / "results").mkdir(parents=True, exist_ok=True)
    (runs_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    steps = []
    
    # Step 1: Forecasting
    if not args.skip_run:
        steps.append((
            "Forecasting (backtest.py --run)",
            [PYTHON, str(SCRIPTS_DIR / "backtest.py"), "--run", "--run-name", run_name, "--limit", str(limit)]
        ))
    
    # Step 2: Grading  
    if not args.skip_grade:
        steps.append((
            "Grading (backtest.py --grade)",
            [PYTHON, str(SCRIPTS_DIR / "backtest.py"), "--grade", "--run-name", run_name]
        ))
    
    # Step 3: Fetch community forecasts via CSV
    steps.append((
        "Fetch Community (fetch_fixed_community.py)",
        [PYTHON, str(SCRIPTS_DIR / "fetch_fixed_community.py"), "--run-name", run_name]
    ))
    
    # Step 4: Generate tables
    steps.append((
        "Generate Tables (gen_tables.py)",
        [PYTHON, str(SCRIPTS_DIR / "gen_tables.py"), "--run-name", run_name]
    ))
    
    # Run all steps
    failed = False
    for step_name, cmd in steps:
        if not run_step(step_name, cmd):
            failed = True
            break
    
    # Summary
    print(f"\n{'#'*60}")
    if failed:
        print(f"# ❌ PIPELINE FAILED")
    else:
        print(f"# ✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"#")
        print(f"# Results: backtesting/data/runs/{run_name}/plots/")
        print(f"#   - binary_results.txt")
        print(f"#   - mc_results.txt")  
        print(f"#   - numeric_results.txt")
    print(f"{'#'*60}\n")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
