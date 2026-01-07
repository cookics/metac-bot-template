import json
import argparse
import sys
from pathlib import Path

# Add root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from visualization import plot_pdf, ensure_plots_dir

def load_run_data(run_name):
    run_dir = ROOT_DIR / "backtesting" / "data" / "runs" / run_name
    results_dir = run_dir / "results"
    
    grades_files = sorted(results_dir.glob("*.grades.json"))
    run_files = sorted(results_dir.glob("run_*.json"))
    run_files = [f for f in run_files if not f.name.endswith('.grades.json')]
    
    if not grades_files or not run_files:
        return None
        
    runs = {}
    for gf, rf in zip(grades_files, run_files):
        with open(gf) as f:
            grades_data = json.load(f)
        with open(rf) as f:
            run_data = json.load(f)
        
        model_name = run_data.get('forecast_model', 'unknown')
        slug = run_data.get('config_name', 'default')
        runs[slug] = {
            "grades": {g['question_id']: g for g in grades_data.get('grades', []) if 'question_id' in g},
            "forecasts": {f['question_id']: f for f in run_data.get('forecasts', []) if 'question_id' in f},
            "model": model_name
        }
    return runs

def main():
    parser = argparse.ArgumentParser(description="Generate multi-model comparison plots")
    parser.add_argument("--runs", type=str, required=True, help="Comma-separated run folders (e.g. backtest_8,backtest_9)")
    parser.add_argument("--primary-slug", type=str, default="default", help="Slug of the 'our' forecast (from backtest_8 usually)")
    parser.add_argument("--compare-slugs", type=str, required=True, help="Comma-separated slugs to compare against")
    args = parser.parse_args()
    
    all_runs = {}
    for run_name in args.runs.split(","):
        data = load_run_data(run_name)
        if data:
            all_runs.update(data)
            
    if args.primary_slug not in all_runs:
        print(f"Error: Primary slug {args.primary_slug} not found in loaded data.")
        sys.exit(1)
        
    primary = all_runs[args.primary_slug]
    compare_slugs = [s.strip() for s in args.compare_slugs.split(",")]
    
    ensure_plots_dir()
    out_dir = ROOT_DIR / "backtesting" / "data" / "runs" / "backtest_9" / "plots" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating comparison plots: {args.primary_slug} vs {compare_slugs}")
    
    num_plotted = 0
    # Find all numeric questions in primary
    for qid, fc in primary["forecasts"].items():
        if fc.get("question_type") != "numeric":
            continue
            
        details = fc.get("question_details", {})
        scaling = details.get("scaling", {})
        res = float(fc.get("resolution", 0)) if fc.get("resolution") not in [None, "annulled"] else 0
        grade_entry = primary["grades"].get(qid, {})
        comm_cdf = grade_entry.get("community_forecast")
        
        extra_cdfs = {}
        for slug in compare_slugs:
            if slug in all_runs and qid in all_runs[slug]["forecasts"]:
                extra_cdfs[f"{slug} ({all_runs[slug]['model'].split('/')[-1]})"] = all_runs[slug]["forecasts"][qid].get("forecast")
        
        plot_path = out_dir / f"compare_{qid}.png"
        plot_pdf(
            cdf=fc.get("forecast"),
            resolution=res,
            range_min=scaling.get("range_min", 0),
            range_max=scaling.get("range_max", 100),
            title=f"Comparison: {fc.get('title', 'Question')[:60]}...",
            save_path=plot_path,
            community_cdf=comm_cdf,
            extra_cdfs=extra_cdfs
        )
        num_plotted += 1
        print(f"  [{num_plotted}/17] Plotted QID {qid}")

    print(f"\nDone! 17 comparison plots saved to {out_dir}")

if __name__ == "__main__":
    main()
