import json
import math
import sys
from pathlib import Path

# Add root to sys.path to import config/api from parent
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

PLOTS_DIR = Path(__file__).resolve().parent.parent / "data" / "plots"
grades_path = Path(__file__).resolve().parent.parent / "data" / "results" / 'run_20260105_234222_clean_run.grades.json'

# Load grades and run data
with open(grades_path) as f:
    grades_data = json.load(f)

# Hardcoded run path for now as per session context
run_path = Path(__file__).resolve().parent.parent / "data" / "results" / 'run_20260105_234222_clean_run.json'
with open(run_path) as f:
    run_data = json.load(f)

title_to_scaling = {}
for f in run_data.get('forecasts', []):
    scaling = f.get('question_details', {}).get('scaling', {})
    title_to_scaling[f.get('title')] = scaling

grades = grades_data.get('grades', [])

def log_score_binary(fc, outcome, eps=0.001):
    fc = max(eps, min(1-eps, fc))
    return math.log(fc) if outcome == 1 else math.log(1-fc)

def peer_score_binary(our_fc, comm_fc, outcome):
    return 100 * (log_score_binary(our_fc, outcome) - log_score_binary(comm_fc, outcome))

def log_score_mc(prob_on_correct, eps=0.001):
    return math.log(max(eps, prob_on_correct))

def peer_score_mc(our_prob, comm_prob, eps=0.001):
    return 100 * (log_score_mc(our_prob, eps) - log_score_mc(comm_prob, eps))

def get_normalized_density(cdf, resolution, range_min, range_max):
    """Estimate normalized PDF density (0-1 space) at resolution point from 201-point CDF."""
    if not cdf or len(cdf) != 201 or resolution is None:
        return 0.0
    if range_max <= range_min:
        return 0.0

    # Normalize resolution to [0, 1]
    res_norm = (resolution - range_min) / (range_max - range_min)
    res_norm = max(0.0, min(1.0, res_norm))

    # Grid index (0 to 200)
    idx = res_norm * 200
    i = int(round(idx))
    
    # Use a small window for stability (3 points)
    low = max(0, i - 1)
    high = min(200, i + 1)
    if high == low: return 0.0
    
    dp = cdf[high] - cdf[low]
    # Corresponding range in unit-less normalized space
    dx_norm = (high - low) / 200.0
    
    # Normalized density (PDF height in unit-range space)
    # Uniform distribution = 1.0, Max = 200.0
    return dp / dx_norm if dx_norm > 0 else 0.0

def crps_from_cdf(cdf, resolution, range_min, range_max):
    pass

# ============= BINARY =============
print("=" * 60)
print("BINARY QUESTIONS")
print("=" * 60)
binary_grades = [g for g in grades if g.get('question_type') == 'binary']

# Recalc peer scores
for g in binary_grades:
    comm = g.get('community_forecast')
    if comm is not None:
        g['peer_score'] = peer_score_binary(g['forecast'], comm, g['outcome'])

with open(PLOTS_DIR / 'binary_results.txt', 'w') as f:
    f.write("=" * 130 + "\n")
    f.write("BINARY QUESTIONS RESULTS\n")
    f.write("=" * 130 + "\n\n")
    f.write(f"{'#':<3} {'Our Pred':<10} {'Comm Pred':<10} {'Outcome':<8} {'Log Score':<12} {'Peer Score':<12} Title\n")
    f.write("-" * 130 + "\n")
    
    for i, g in enumerate(binary_grades, 1):
        our = g.get('forecast', 0)
        comm = g.get('community_forecast')
        comm_str = f"{comm:.3f}" if comm is not None else "N/A"
        outcome = g.get('outcome', 0)
        log_score = g.get('log_score', 0)
        peer = g.get('peer_score')
        peer_str = f"{peer:.2f}" if peer is not None else "N/A"
        title = g.get('title', '')[:65]
        correct = (our > 0.5 and outcome == 1) or (our < 0.5 and outcome == 0)
        marker = "" if correct else " *** WRONG"
        f.write(f"{i:<3} {our:<10.3f} {comm_str:<10} {outcome:<8} {log_score:<12.4f} {peer_str:<12} {title}{marker}\n")
    
    f.write("-" * 130 + "\n")
    f.write(f"\nSUMMARY: {len(binary_grades)} binary questions\n")
    correct_count = sum(1 for g in binary_grades if (g['forecast'] > 0.5 and g['outcome'] == 1) or (g['forecast'] < 0.5 and g['outcome'] == 0))
    f.write(f"Correct predictions: {correct_count}/{len(binary_grades)} ({100*correct_count/len(binary_grades):.1f}%)\n")
    peer_scores = [g['peer_score'] for g in binary_grades if g.get('peer_score') is not None]
    if peer_scores:
        f.write(f"Mean Peer Score: {sum(peer_scores)/len(peer_scores):.2f} (n={len(peer_scores)}) (positive = beating community)\n")
    f.write(f"Mean Log Score: {sum(g['log_score'] for g in binary_grades)/len(binary_grades):.4f}\n")

print(f"Binary: {len(binary_grades)} questions, {len(peer_scores)} with peer scores")

# ============= MULTIPLE CHOICE =============
print("\n" + "=" * 60)
print("MULTIPLE CHOICE QUESTIONS")
print("=" * 60)
mc_grades = [g for g in grades if g.get('question_type') == 'multiple_choice']

# Calc peer scores for MC
for g in mc_grades:
    comm_fc = g.get('community_forecast', {})
    our_fc = g.get('forecast', {})
    resolution = g.get('resolution', '')
    
    # If community forecast is a list (common from CSV), map to options
    if isinstance(comm_fc, list):
        # Find options from run_data
        options = []
        for fc in run_data.get('forecasts', []):
            if fc.get('title') == g.get('title'):
                options = fc.get('question_details', {}).get('options', [])
                break
        if options and len(options) == len(comm_fc):
            comm_fc = dict(zip(options, comm_fc))
            g['community_forecast'] = comm_fc
        else:
            comm_fc = {}

    our_prob = our_fc.get(resolution, 0) if our_fc else 0
    comm_prob = comm_fc.get(resolution, 0) if comm_fc else 0
    
    g['correct_probability'] = our_prob
    g['community_correct_probability'] = comm_prob
    
    if comm_prob > 0:
        g['peer_score'] = peer_score_mc(our_prob, comm_prob)

with open(PLOTS_DIR / 'mc_results.txt', 'w') as f:
    f.write("=" * 140 + "\n")
    f.write("MULTIPLE CHOICE QUESTIONS RESULTS\n")
    f.write("=" * 140 + "\n\n")
    
    for i, g in enumerate(mc_grades, 1):
        title = g.get('title', '')[:80]
        resolution = g.get('resolution', '')
        our_fc = g.get('forecast', {})
        comm_fc = g.get('community_forecast', {})
        our_prob = g.get('correct_probability', 0)
        comm_prob = g.get('community_correct_probability', 0)
        log_score = g.get('log_score', 0)
        brier = g.get('brier_score', 0)
        peer = g.get('peer_score')
        
        f.write(f"\n[{i}] {title}\n")
        f.write("-" * 110 + "\n")
        f.write(f"Resolution: {resolution}\n")
        comm_prob_str = f"{comm_prob:.3f}" if comm_prob else "N/A"
        f.write(f"Our prob on correct: {our_prob:.3f} | Community: {comm_prob_str}")
        if peer is not None:
            f.write(f" | Peer Score: {peer:.2f}")
        f.write(f"\nLog Score: {log_score:.4f} | Brier: {brier:.4f}\n\n")

        f.write(f"{'Option':<55} {'Us':<10} {'Comm':<10} {'Correct?'}\n")
        f.write("-" * 90 + "\n")

        all_opts = sorted(set(our_fc.keys()) | set(comm_fc.keys() if comm_fc else []))
        for opt in all_opts:
            our_p = our_fc.get(opt, 0)
            comm_p = comm_fc.get(opt) if comm_fc else None
            comm_str = f"{comm_p:.3f}" if comm_p is not None else "N/A"
            is_correct = "  <-- CORRECT" if str(opt) == str(resolution) else ""
            f.write(f"{str(opt)[:54]:<55} {our_p:<10.3f} {comm_str:<10} {is_correct}\n")
    
    f.write("\n" + "=" * 140 + "\n")
    f.write(f"SUMMARY: {len(mc_grades)} multiple choice questions\n")
    peer_scores = [g['peer_score'] for g in mc_grades if g.get('peer_score') is not None]
    if peer_scores:
        f.write(f"Mean Peer Score: {sum(peer_scores)/len(peer_scores):.2f} (n={len(peer_scores)})\n")
    f.write(f"Mean Log Score: {sum(g['log_score'] for g in mc_grades)/len(mc_grades):.4f}\n")
    f.write(f"Mean Brier Score: {sum(g['brier_score'] for g in mc_grades)/len(mc_grades):.4f}\n")

print(f"MC: {len(mc_grades)} questions, {len(peer_scores)} with peer scores")

# ============= NUMERIC =============
print("\n" + "=" * 60)
print("NUMERIC QUESTIONS")
print("=" * 60)
num_grades = [g for g in grades if g.get('question_type') == 'numeric']

# Calc peer Density/Log scores for numeric
for g in num_grades:
    title = g.get('title')
    comm_cdf = g.get('community_forecast')
    # Find the CDF and scaling in the original run data
    for fc in run_data.get('forecasts', []):
        if fc.get('title') == title:
            our_cdf = fc.get('forecast')
            details = fc.get('question_details', {})
            scaling = details.get('scaling', {})
            range_min = scaling.get('range_min', 0)
            range_max = scaling.get('range_max', 1)
            break
    else:
        our_cdf = None
        range_min, range_max = 0, 1

    resolution = g.get('resolution')

    if comm_cdf and isinstance(comm_cdf, list) and len(comm_cdf) == 201 and resolution is not None:
        # Calculate normalized densities (0-1 range space)
        comm_norm_den = get_normalized_density(comm_cdf, resolution, range_min, range_max)
        our_norm_den = get_normalized_density(our_cdf, resolution, range_min, range_max) if our_cdf else 0

        g['community_density_norm'] = comm_norm_den
        g['our_density_norm'] = our_norm_den

        if comm_norm_den > 0:
            # Numeric Log Score = ln(normalized_density)
            # Uniform = ln(1) = 0. Use 1e-5 as floor (very conservative)
            our_log = math.log(max(0.00001, our_norm_den))
            comm_log = math.log(max(0.00001, comm_norm_den))
            g['peer_score'] = 100 * (our_log - comm_log)
            g['log_score'] = our_log
            g['community_log_score'] = comm_log

with open(PLOTS_DIR / 'numeric_results.txt', 'w') as f:
    f.write("=" * 130 + "\n")
    f.write("NUMERIC QUESTIONS RESULTS (Normalized Density scoring)\n")
    f.write("=" * 130 + "\n\n")
    f.write(f"{'#':<3} {'Resolution':<14} {'Our NormD':<10} {'Comm NormD':<10} {'Our Log':<10} {'Peer':<10} Title\n")
    f.write("-" * 130 + "\n")

    for i, g in enumerate(num_grades, 1):
        res = g.get('resolution', 0)
        our_den = g.get('our_density_norm', 0)
        comm_den = g.get('community_density_norm', 0)
        our_log = g.get('log_score', 0)
        peer = g.get('peer_score')
        peer_str = f"{peer:.2f}" if peer is not None else "N/A"
        title = g.get('title', '')[:65]

        # Use scientific notation for resolution if large, standard for density
        res_str = f"{res:.4g}" if abs(res) > 1e6 else f"{res:.4f}"
        f.write(f"{i:<3} {res_str:<14} {our_den:<10.3f} {comm_den:<10.3f} {our_log:<10.4f} {peer_str:<10} {title}\n")

    f.write("-" * 130 + "\n")
    f.write(f"\nSUMMARY: {len(num_grades)} numeric questions\n")
    log_scores = [g['log_score'] for g in num_grades if g.get('log_score') is not None]
    if log_scores:
        f.write(f"Mean Log Score: {sum(log_scores)/len(log_scores):.4f} (Uniform = 0.0)\n")
    peer_scores = [g['peer_score'] for g in num_grades if g.get('peer_score') is not None]
    if peer_scores:
        f.write(f"Mean Peer Score: {sum(peer_scores)/len(peer_scores):.2f} (n={len(peer_scores)}) (positive = beating community)\n")

print(f"Numeric: {len(num_grades)} questions, {len(peer_scores)} with peer scores")

# Save updated grades
with open(grades_path, 'w') as f:
    json.dump(grades_data, f, indent=2)
print(f"\nSaved updated grades to {grades_path}")
print(f"Tables written to {PLOTS_DIR}")
