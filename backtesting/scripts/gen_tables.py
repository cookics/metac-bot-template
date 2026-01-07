import json
import math
import sys
import argparse
from pathlib import Path

# Add root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

def load_run_data_from_file(grades_file):
    results_dir = grades_file.parent
    
    # Matching run file usually starts with "run_" and has same timestamp
    run_file_name = grades_file.name.replace(".grades.json", ".json")
    run_file = results_dir / run_file_name
    
    if not run_file.exists():
        # Fallback to finding the nearest run_*.json that isn't a grade
        run_files = [f for f in results_dir.glob("run_*.json") if not f.name.endswith(".grades.json")]
        if not run_files:
            return None
        run_file = run_files[0]
        
    with open(grades_file) as f:
        grades_data = json.load(f)
    with open(run_file) as f:
        run_data = json.load(f)
        
    return {
        "run_file": grades_file.name,
        "grades": grades_data.get('grades', []),
        "run_info": run_data
    }

def main():
    parser = argparse.ArgumentParser(description="Generate results tables and comparisons")
    parser.add_argument("--run-name", type=str, required=True, help="Name of the primary run folder")
    parser.add_argument("--compare-runs", type=str, help="Comma-separated list of other run names to include")
    args = parser.parse_args()
    
    RUN_DIR = Path(__file__).resolve().parent.parent / "data" / "runs" / args.run_name
    RESULTS_DIR = RUN_DIR / "results"
    PLOTS_DIR = RUN_DIR / "plots"
    
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory {RESULTS_DIR} does not exist.")
        sys.exit(1)
        
    grades_files = sorted(RESULTS_DIR.glob("*.grades.json"))
    
    if not grades_files:
        print(f"No grades found in {RESULTS_DIR}")
        sys.exit(1)
        
    all_runs = []
    
    # Process primary run
    for gf in sorted(RESULTS_DIR.glob("*.grades.json")):
        data = load_run_data_from_file(gf)
        if data:
            all_runs.append(data)
            
    # Process additional comparison runs
    if args.compare_runs:
        for other_run_name in args.compare_runs.split(","):
            other_run_name = other_run_name.strip()
            other_results_dir = RUN_DIR.parent / other_run_name / "results"
            if other_results_dir.exists():
                for gf in sorted(other_results_dir.glob("*.grades.json")):
                    data = load_run_data_from_file(gf)
                    if data:
                        all_runs.append(data)
            else:
                print(f"Warning: Comparison run directory {other_results_dir} not found.")

    print(f"Found {len(all_runs)} graded models total.")
    
    # 1. Update individual results tables for each run based on latest grading.py logic
    from grading import generate_detailed_tables
    for r in all_runs:
        tables = generate_detailed_tables(r["grades"], r["run_info"])
        # We might want to prefix these with the model name to avoid overwriting if they use the same plots dir
        # But usually each run_full_backtest.py run points to the same PLOTS_DIR.
        # Let's save them as [model]_binary_results.txt etc.
        model_slug = r["run_info"].get("config_name", "default")
        for filename, content in tables.items():
            out_name = f"{model_slug}_{filename}"
            with open(PLOTS_DIR / out_name, 'w') as f:
                f.write(content)
        print(f"  [{model_slug}] Updated results tables.")

    # 2. Generate the Comparison Summary
    if len(all_runs) >= 1:
        comp_file = PLOTS_DIR / "comparison_summary.txt"
        
        # Sort runs by model name for consistency
        all_runs.sort(key=lambda x: x['run_info'].get('forecast_model', 'unknown'))
        
        with open(comp_file, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("BACKTEST MULTI-MODEL COMPARISON SUMMARY\n")
            f.write("=" * 120 + "\n\n")
            
            # Header
            header = f"{'Category':<20}"
            for r in all_runs:
                model = r['run_info'].get('forecast_model', 'unknown').split('/')[-1]
                slug = r['run_info'].get('config_name', 'default')
                header += f" | {slug:<25}"
            f.write(header + "\n")
            
            header_models = f"{'Model':<20}"
            for r in all_runs:
                model = r['run_info'].get('forecast_model', 'unknown').split('/')[-1]
                header_models += f" | {model[:25]:<25}"
            f.write(header_models + "\n")
            f.write("-" * 120 + "\n")
            
            # Binary Accuracy
            f.write(f"{'Binary Accuracy':<20}")
            for r in all_runs:
                bg = [g for g in r['grades'] if g.get('question_type') == 'binary']
                if bg:
                    acc = sum(1 for g in bg if (g.get('forecast', 0) > 0.5 and g.get('outcome') == 1) or (g.get('forecast', 0) < 0.5 and g.get('outcome') == 0)) / len(bg)
                    f.write(f" | {acc*100:>6.1f}% ({len(bg)}q)          ")
                else:
                    f.write(f" | {'N/A':<25}")
            f.write("\n")
            
            # Peer Scores
            for cat in ['binary', 'multiple_choice', 'numeric']:
                f.write(f"{cat.capitalize() + ' Avg Peer':<20}")
                for r in all_runs:
                    cg = [g for g in r['grades'] if g.get('question_type') == cat]
                    peer_scores = [g.get('peer_score') for g in cg if g.get('peer_score') is not None]
                    if peer_scores:
                        avg = sum(peer_scores) / len(peer_scores)
                        f.write(f" | {avg:>10.2f}                ")
                    else:
                        f.write(f" | {'N/A':<25}")
                f.write("\n")
            
            f.write("-" * 120 + "\n")
            f.write(f"{'OVERALL AVG PEER':<20}")
            for r in all_runs:
                all_peers = [g.get('peer_score') for g in r['grades'] if g.get('peer_score') is not None]
                if all_peers:
                    avg = sum(all_peers) / len(all_peers)
                    f.write(f" | {avg:>10.2f}                ")
                else:
                    f.write(f" | {'N/A':<25}")
            f.write("\n")

        print(f"Final comparison table saved to {comp_file}")

if __name__ == "__main__":
    main()
