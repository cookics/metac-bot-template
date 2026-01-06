"""Regenerate ALL backtest plots including CDFs and summary visualizations."""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add root to sys.path to import config/api from parent
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from grading import grade_numeric_forecast, get_resolution_idx
from visualization import plot_cdf, generate_all_plots

PLOTS_DIR = Path(__file__).resolve().parent.parent / "data" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load grades and run data
grades_path = Path(__file__).resolve().parent.parent / "data" / "results" / 'run_20260105_234222_clean_run.grades.json'
with open(grades_path) as f:
    grades_data = json.load(f)

run_path = Path(__file__).resolve().parent.parent / "data" / "results" / 'run_20260105_234222_clean_run.json'
with open(run_path) as f:
    run_data = json.load(f)

title_to_forecast = {f.get('title'): f for f in run_data.get('forecasts', [])}
grades = grades_data.get('grades', [])

# Filter numeric for individual CDFs
num_grades = [g for g in grades if g.get('question_type') == 'numeric']
print(f"Found {len(num_grades)} numeric questions for CDF plotting")

for i, g in enumerate(num_grades, 1):
    title = g.get('title', '')
    
    # Get our forecast CDF from run data
    original_forecast = title_to_forecast.get(title, {})
    fc = original_forecast.get('forecast')
    
    comm = g.get('community_forecast')
    
    # Metadata for plotting
    resolution = g.get('resolution')
    
    # Get scaling from original forecast
    details = original_forecast.get('question_details', {})
    scaling = details.get('scaling', {})
    range_min = scaling.get('range_min')
    range_max = scaling.get('range_max')
    
    if not fc or range_min is None or range_max is None:
        print(f"Skipping {i}: Missing data. fc={len(fc) if fc else 'None'}, r_min={range_min}, r_max={range_max}")
        continue

    # Filename
    # Use post_id if available, otherwise just index
    post_id = details.get('post_id', f"q{i}")
    save_path = PLOTS_DIR / f"cdf_{i:02d}_{post_id}.png"
    
    print(f"Plotting {i}: {title[:40]} -> {save_path.name}")
    
    try:
        plot_cdf(
            cdf=fc,
            resolution=float(resolution) if resolution is not None else (range_min + range_max)/2,
            range_min=range_min,
            range_max=range_max,
            title=f"Q{i}: {title[:30]}...",
            save_path=save_path,
            community_cdf=comm if (comm and len(comm) == 201) else None
        )
    except Exception as e:
        print(f"Error plotting {i}: {e}")

# Regenerate all summary plots (categorical, numeric, scores)
print("\nRegenerating summary plots...")
try:
    generate_all_plots(grades, run_data.get('forecasts', []))
except Exception as e:
    print(f"Error generating summary plots: {e}")

print("\nDone building all plots!")
