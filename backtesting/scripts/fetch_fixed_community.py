import json
import requests
import zipfile
import io
import csv
import math
import sys
from pathlib import Path
# Add root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))
from config import AUTH_HEADERS, API_BASE_URL

# PATHS
grades_path = ROOT_DIR / 'backtesting' / 'data' / 'runs' / 'backtest_2' / 'results' / 'run_20260106_014406_default.grades.json'

def fetch_group_community(post_id, question_id):
    """Fetch community forecast from CSV download for group sub-questions."""
    url = f"{API_BASE_URL}/posts/{post_id}/download-data/?sub_question={question_id}"
    print(f"  Downloading CSV for Q{question_id} in Post {post_id}...")
    
    response = requests.get(url, **AUTH_HEADERS)
    if not response.ok:
        print(f"    Failed: {response.status_code}")
        return None
        
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Look for forecast_data.csv
            csv_name = 'forecast_data.csv'
            if csv_name not in z.namelist():
                # Try finding any csv
                csv_files = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_files: return None
                csv_name = csv_files[0]
                
            with z.open(csv_name) as f:
                content = f.read().decode('utf-8')
                reader = csv.DictReader(io.StringIO(content))
                rows = list(reader)
                
                # We want the last aggregation row (unweighted or recency_weighted)
                # Filter for "Forecaster Username" == "unweighted" or "recency_weighted"
                agg_rows = [r for r in rows if r.get('Forecaster Username') in ['unweighted', 'recency_weighted']]
                if not agg_rows:
                    return None
                
                # Take the latest one
                target = agg_rows[-1]
                
                # Depending on question type, extract probability or CDF
                # For Binary/MC: "Probability Yes Per Category"
                # For Numeric: "Continuous CDF"
                
                if target.get('Probability Yes'):
                    try:
                        return float(target.get('Probability Yes'))
                    except:
                        pass
                
                if target.get('Probability Yes Per Category'):
                    try:
                        # Value looks like "{'yes': 0.85}" or "{'A': 0.1, 'B': 0.9}"
                        val_str = target.get('Probability Yes Per Category').replace("'", '"')
                        probs = json.loads(val_str)
                        # If simple binary, return the 'yes' float
                        if 'yes' in probs and len(probs) == 1:
                            return probs['yes']
                        return probs
                    except:
                        return None
                        
                if target.get('Continuous CDF'):
                    try:
                        val_str = target.get('Continuous CDF')
                        # CDF is a list of 201 floats
                        cdf = json.loads(val_str)
                        if isinstance(cdf, list) and len(cdf) == 201:
                            return cdf
                    except:
                        return None
    except Exception as e:
        print(f"    Error parsing ZIP/CSV: {e}")
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch community forecasts via CSV API")
    parser.add_argument("--run-name", type=str, default="backtest_3", help="Name of the run folder")
    args = parser.parse_args()
    
    run_name = args.run_name
    results_dir = Path(__file__).resolve().parent.parent / "data" / "runs" / run_name / "results"
    
    # Auto-detect latest grades file
    grades_files = sorted(results_dir.glob("*.grades.json"), reverse=True)
    run_files = sorted(results_dir.glob("run_*.json"), reverse=True)
    run_files = [f for f in run_files if not f.name.endswith('.grades.json')]
    
    if not grades_files:
        print(f"No grades files found in {results_dir}")
        return
    if not run_files:
        print(f"No run files found in {results_dir}")
        return
        
    grades_file = grades_files[0]
    run_file = run_files[0]
    
    print(f"[{run_name}] Using grades: {grades_file.name}")
    print(f"[{run_name}] Using run: {run_file.name}")

    with open(run_file, 'r', encoding='utf-8') as f:
        run_data = json.load(f)
    
    # Map title to (post_id, question_id)
    title_to_ids = {}
    for fc in run_data.get('forecasts', []):
        title_to_ids[fc.get('title')] = (
            fc.get('question_id'),
            fc.get('question_details', {}).get('post_id')
        )

    with open(grades_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    grades = data['grades']
    updated_count = 0
    
    for i, g in enumerate(grades):
        # Even if it has a dummy value (like 0 in peer score), we might want to refresh if community is missing
        if g.get('community_forecast') is not None:
             continue
             
        title = g.get('title', 'Unknown')
        q_id, p_id = title_to_ids.get(title, (None, None))
        
        if not p_id or not q_id:
            print(f"[{i+1}] Missing IDs for: {title[:40]}")
            continue
            
        print(f"[{i+1}/{len(grades)}] Fetching community for: {title[:40]}...")
        comm = fetch_group_community(p_id, q_id)
        
        if comm is not None:
            g['community_forecast'] = comm
            updated_count += 1
            print(f"    SUCCESS: Fetched community forecast.")
        else:
            print(f"    STILL MISSING.")
        
        import time
        time.sleep(1.5)  # Avoid 429

    if updated_count > 0:
        with open(grades_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nUpdated {updated_count} community forecasts in {grades_file.name}")
    else:
        print("\nNo community forecasts needed updating (all already present).")

if __name__ == "__main__":
    main()

