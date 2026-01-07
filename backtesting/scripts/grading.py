"""
Grading system for backtesting - calculates forecast accuracy scores.

Supports:
- Log Score (all types, using PDF for continuous)
- Brier Score (binary, multiple choice)
- Peer Score (vs community forecast)
- Baseline Score (vs uniform distribution)
- CRPS (for continuous questions)
"""
import math
import sys
from pathlib import Path
from typing import Union

# Add root and src to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

# Minimum PDF value (sharper floor for 200 bins)
MIN_PDF = 0.0001


def brier_score(forecast: float, outcome: int) -> float:
    """Brier score for a binary forecast. 0 = perfect, 1 = worst."""
    return (forecast - outcome) ** 2


def log_score_binary(forecast: float, outcome: int, epsilon: float = 0.001) -> float:
    """Log score for a binary forecast. More negative = worse."""
    forecast = max(epsilon, min(1 - epsilon, forecast))
    if outcome == 1:
        return math.log(forecast)
    else:
        return math.log(1 - forecast)


# ========================= CONTINUOUS LOG SCORE =========================

def cdf_to_pdf(cdf: list[float]) -> list[float]:
    """
    Convert a CDF to PDF by taking differences.
    
    Args:
        cdf: 201-point CDF (cumulative probabilities)
    
    Returns:
        200-point PDF (probability density at each bin)
    """
    pdf = []
    for i in range(1, len(cdf)):
        density = cdf[i] - cdf[i-1]
        # Apply minimum floor like Metaculus
        pdf.append(max(MIN_PDF, density))
    return pdf


def log_score_continuous(cdf: list[float], resolution_idx: int) -> float:
    """
    Log score for a continuous forecast using PDF at outcome.
    
    ln(pdf(outcome)) where pdf is the probability density at the outcome.
    Can be positive (sharp correct prediction) or negative.
    
    Args:
        cdf: 201-point CDF
        resolution_idx: Index where resolution falls (0-200)
    
    Returns:
        Log score (can be positive or negative)
    """
    pdf = cdf_to_pdf(cdf)
    
    # Get PDF value at resolution (use index-1 since PDF has 200 pts)
    pdf_idx = max(0, min(len(pdf) - 1, resolution_idx - 1))
    pdf_at_outcome = pdf[pdf_idx]
    
    return math.log(max(MIN_PDF, pdf_at_outcome))


# ========================= BASELINE SCORE =========================

def baseline_score_binary(forecast: float, outcome: int) -> float:
    """
    Baseline score for binary: comparison to 50% chance.
    Positive = better than chance, Negative = worse.
    Scaled so perfect prediction ≈ +100.
    
    Formula: 100 * (ln(p) - ln(0.5)) for correct outcome
    """
    baseline_log = log_score_binary(0.5, outcome)
    forecast_log = log_score_binary(forecast, outcome)
    
    # Scale factor so perfect = ~100
    return 100 * (forecast_log - baseline_log)


def baseline_score_continuous(cdf: list[float], resolution_idx: int) -> float:
    """
    Baseline score for continuous: comparison to uniform distribution.
    Positive = better than uniform, Negative = worse.
    
    Uniform CDF has PDF = 1/201 ≈ 0.00498, floored to 0.01
    """
    our_log_score = log_score_continuous(cdf, resolution_idx)
    
    # Uniform baseline (flat PDF)
    uniform_pdf = 1.0 / 200
    baseline_log_score = math.log(max(MIN_PDF, uniform_pdf))
    
    # Scale: continuous empirical max is ~183, so use factor to normalize
    return 100 * (our_log_score - baseline_log_score)


def baseline_score_mc(forecast: dict, resolution: str) -> float:
    """Baseline score for multiple choice: comparison to 1/N uniform."""
    n_options = len(forecast)
    if n_options == 0:
        return 0
    
    uniform_prob = 1.0 / n_options
    correct_prob = forecast.get(resolution, 0)
    
    our_log = math.log(max(0.001, correct_prob))
    baseline_log = math.log(uniform_prob)
    
    return 100 * (our_log - baseline_log)


# ========================= PEER SCORE =========================

def peer_score_binary(our_forecast: float, community_forecast: float, outcome: int) -> float:
    """
    Peer score for binary: how much better/worse than community.
    
    Formula: 100 * (our_log - community_log)
    """
    our_log = log_score_binary(our_forecast, outcome)
    community_log = log_score_binary(community_forecast, outcome)
    
    return 100 * (our_log - community_log)


def peer_score_continuous(our_cdf: list[float], community_cdf: list[float], resolution_idx: int) -> float:
    """
    Peer score for continuous: how much better/worse than community.
    
    Formula: 100 * (our_log - community_log) / 2
    (Divided by 2 per Metaculus scoring rules)
    """
    our_log = log_score_continuous(our_cdf, resolution_idx)
    community_log = log_score_continuous(community_cdf, resolution_idx)
    
    # Continuous peer scores divided by 2 per Metaculus
    return 100 * (our_log - community_log) / 2


def peer_score_mc(our_forecast: dict, community_forecast: dict, resolution: str) -> float:
    """Peer score for multiple choice."""
    our_prob = our_forecast.get(resolution, 0.001)
    community_prob = community_forecast.get(resolution, 0.001) if community_forecast else 0.001
    
    our_log = math.log(max(0.001, our_prob))
    community_log = math.log(max(0.001, community_prob))
    
    return 100 * (our_log - community_log)


# ========================= CRPS (for continuous) =========================

def calculate_crps(cdf: list[float], resolution_idx: int) -> float:
    """CRPS score - lower is better."""
    n = len(cdf)
    crps = 0.0
    for i in range(n):
        heaviside = 1.0 if i >= resolution_idx else 0.0
        crps += (cdf[i] - heaviside) ** 2
    return crps / n


def generate_naive_cdf(n_points: int = 201, tail_prob: float = 0.05) -> list[float]:
    """Naive uniform CDF for baseline comparison."""
    return [tail_prob + (1 - 2 * tail_prob) * i / (n_points - 1) for i in range(n_points)]


def get_normalized_density(cdf: list[float], resolution: float, range_min: float, range_max: float) -> float:
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


# ========================= GRADING FUNCTIONS =========================

def grade_binary_forecast(forecast: float, resolution: bool, community_forecast: float = None) -> dict:
    """Grade a binary forecast with all score types."""
    outcome = 1 if resolution else 0
    
    result = {
        "question_type": "binary",
        "forecast": forecast,
        "outcome": outcome,
        "brier_score": brier_score(forecast, outcome),
        "log_score": log_score_binary(forecast, outcome),
        "baseline_score": baseline_score_binary(forecast, outcome),
    }
    
    if community_forecast is not None:
        result["community_forecast"] = community_forecast
        result["peer_score"] = peer_score_binary(forecast, community_forecast, outcome)
    
    return result


def grade_multiple_choice_forecast(forecast: dict, resolution: str, community_forecast: dict = None) -> dict:
    """Grade a multiple choice forecast."""
    if not forecast or not resolution:
        return {"error": "Missing forecast or resolution"}
    
    correct_prob = forecast.get(resolution, 0)
    n_options = len(forecast)
    
    # Brier-like
    brier = (1 - correct_prob) ** 2
    
    result = {
        "question_type": "multiple_choice",
        "forecast": forecast,
        "resolution": resolution,
        "correct_probability": correct_prob,
        "brier_score": brier,
        "log_score": math.log(max(0.001, correct_prob)),
        "baseline_score": baseline_score_mc(forecast, resolution),
    }
    
    if community_forecast and isinstance(community_forecast, dict):
        result["community_forecast"] = community_forecast
        result["peer_score"] = peer_score_mc(forecast, community_forecast, resolution)
    
    return result


def get_resolution_idx(resolution: float, range_min: float, range_max: float) -> int:
    """Calculate resolution index for a continuous question."""
    if resolution <= range_min:
        return 0
    elif resolution >= range_max:
        return 200
    else:
        step = (range_max - range_min) / 200
        return min(200, max(0, int((resolution - range_min) / step)))


def grade_numeric_forecast(
    cdf: list[float],
    resolution: float,
    range_min: float,
    range_max: float,
    community_cdf: list[float] = None
) -> dict:
    """Grade a numeric/continuous forecast with all score types."""
    if not cdf or len(cdf) != 201:
        return {"error": f"Invalid CDF length: {len(cdf) if cdf else 0}"}
    
    resolution_idx = get_resolution_idx(resolution, range_min, range_max)
    
    # CRPS and baselines
    forecast_crps = calculate_crps(cdf, resolution_idx)
    naive_cdf = generate_naive_cdf(201)
    baseline_crps = calculate_crps(naive_cdf, resolution_idx)
    
    # Skill score (improvement over naive)
    skill_score = 1 - (forecast_crps / baseline_crps) if baseline_crps > 0 else 0
    
    # Calculate normalized densities for table reporting (not used directly for Peer score math, but helpful)
    our_norm_den = get_normalized_density(cdf, resolution, range_min, range_max)
    
    result = {
        "question_type": "numeric",
        "resolution": resolution,
        "resolution_idx": resolution_idx,
        "predicted_percentile": cdf[resolution_idx],
        "crps": forecast_crps,
        "log_score": log_score_continuous(cdf, resolution_idx),
        "our_density_norm": our_norm_den,
        "baseline_score": baseline_score_continuous(cdf, resolution_idx),
        "skill_score": skill_score,
        "brier_score": forecast_crps,  # For aggregate compatibility
    }
    
    if community_cdf and len(community_cdf) == 201:
        comm_norm_den = get_normalized_density(community_cdf, resolution, range_min, range_max)
        result["community_forecast"] = community_cdf
        result["community_density_norm"] = comm_norm_den
        result["peer_score"] = peer_score_continuous(cdf, community_cdf, resolution_idx)
        result["community_crps"] = calculate_crps(community_cdf, resolution_idx)
    
    return result


def grade_forecast(
    forecast: Union[float, list[float], dict],
    resolution: Union[bool, float, str],
    question_type: str,
    question_details: dict = None,
    community_forecast: Union[float, list[float], dict] = None
) -> dict:
    """
    Unified grading function for all question types.
    
    Args:
        forecast: Our forecast
        resolution: Actual resolution
        question_type: "binary", "numeric", or "multiple_choice"
        question_details: For numeric questions (scaling info)
        community_forecast: Community forecast for Peer Score
    """
    # Handle annulled/ambiguous
    if resolution in ["annulled", "ambiguous", None, ""]:
        return {"error": f"Cannot grade: resolution is {resolution}"}
    
    if question_type == "binary":
        if resolution not in ["yes", "no", True, False, 1, 0]:
            return {"error": f"Invalid binary resolution: {resolution}"}
        resolved_yes = resolution in ["yes", True, 1]
        community = community_forecast if isinstance(community_forecast, (int, float)) else None
        return grade_binary_forecast(forecast, resolved_yes, community)
    
    elif question_type == "numeric":
        if not question_details:
            return {"error": "Numeric question requires question_details with scaling"}
        
        try:
            resolution_value = float(resolution)
        except (ValueError, TypeError):
            return {"error": f"Cannot convert resolution to float: {resolution}"}
        
        scaling = question_details.get("scaling", {})
        community = community_forecast if isinstance(community_forecast, list) else None
        
        return grade_numeric_forecast(
            forecast,
            resolution_value,
            scaling.get("range_min", 0),
            scaling.get("range_max", 100),
            community
        )
    
    elif question_type == "multiple_choice":
        # Handle community forecast in either dict or list format
        community = None
        
        # First, extract the community forecast data (might be in a wrapper dict or raw)
        cf = community_forecast
        if isinstance(cf, dict) and "probability_yes_per_category" in cf:
            cf = cf["probability_yes_per_category"]
        
        if isinstance(cf, dict):
            # Already in dict format - use directly
            community = cf
        elif isinstance(cf, list) and isinstance(forecast, dict):
            # List format from CSV - convert to dict using our forecast's keys
            # This assumes the list order matches the option order
            options = list(forecast.keys())
            if len(cf) == len(options):
                community = dict(zip(options, cf))
        
        return grade_multiple_choice_forecast(forecast, resolution, community)
    
    return {"error": f"Unknown question type: {question_type}"}


# ========================= AGGREGATE SCORING =========================

def calculate_aggregate_scores(grades: list[dict]) -> dict:
    """Calculate aggregate scores across multiple forecasts."""
    results = {
        "total_graded": len([g for g in grades if "error" not in g]),
    }
    
    # Binary
    binary_grades = [g for g in grades if g.get("question_type") == "binary"]
    if binary_grades:
        results["binary"] = {
            "n": len(binary_grades),
            "mean_brier": sum(g["brier_score"] for g in binary_grades) / len(binary_grades),
            "mean_log_score": sum(g["log_score"] for g in binary_grades) / len(binary_grades),
            "mean_baseline": sum(g["baseline_score"] for g in binary_grades) / len(binary_grades),
        }
        peer_scores = [g.get("peer_score") for g in binary_grades if "peer_score" in g]
        if peer_scores:
            results["binary"]["mean_peer"] = sum(peer_scores) / len(peer_scores)
    
    # Multiple choice
    mc_grades = [g for g in grades if g.get("question_type") == "multiple_choice"]
    if mc_grades:
        results["multiple_choice"] = {
            "n": len(mc_grades),
            "mean_brier": sum(g["brier_score"] for g in mc_grades) / len(mc_grades),
            "mean_baseline": sum(g["baseline_score"] for g in mc_grades) / len(mc_grades),
        }
        peer_scores = [g.get("peer_score") for g in mc_grades if "peer_score" in g]
        if peer_scores:
            results["multiple_choice"]["mean_peer"] = sum(peer_scores) / len(peer_scores)
    
    # Numeric
    numeric_grades = [g for g in grades if g.get("question_type") == "numeric"]
    if numeric_grades:
        results["numeric"] = {
            "n": len(numeric_grades),
            "mean_crps": sum(g["crps"] for g in numeric_grades) / len(numeric_grades),
            "mean_log_score": sum(g["log_score"] for g in numeric_grades) / len(numeric_grades),
            "mean_baseline": sum(g["baseline_score"] for g in numeric_grades) / len(numeric_grades),
            "mean_skill": sum(g["skill_score"] for g in numeric_grades) / len(numeric_grades),
        }
        peer_scores = [g.get("peer_score") for g in numeric_grades if "peer_score" in g]
        if peer_scores:
            results["numeric"]["mean_peer"] = sum(peer_scores) / len(peer_scores)
    
    return results


def generate_report(grades: list[dict], config_name: str = "default") -> str:
    """Generate a human-readable report."""
    agg = calculate_aggregate_scores(grades)
    
    report = f"""
# Backtest Report: {config_name}

## Summary ({agg.get('total_graded', 0)} forecasts graded)
"""
    
    if "binary" in agg:
        b = agg["binary"]
        report += f"""
### Binary (n={b['n']})
| Metric | Score |
|--------|-------|
| Mean Brier | {b['mean_brier']:.4f} |
| Mean Log Score | {b['mean_log_score']:.4f} |
| Mean Baseline | {b['mean_baseline']:+.2f} |
"""
        if "mean_peer" in b:
            report += f"| Mean Peer | {b['mean_peer']:+.2f} |\n"
    
    if "numeric" in agg:
        n = agg["numeric"]
        report += f"""
### Numeric (n={n['n']})
| Metric | Score |
|--------|-------|
| Mean CRPS | {n['mean_crps']:.4f} |
| Mean Log Score | {n['mean_log_score']:.4f} |
| Mean Baseline | {n['mean_baseline']:+.2f} |
| Mean Skill | {n['mean_skill']:.2%} |
"""
        if "mean_peer" in n:
            report += f"| Mean Peer | {n['mean_peer']:+.2f} |\n"
    
    if "multiple_choice" in agg:
        m = agg["multiple_choice"]
        report += f"""
### Multiple Choice (n={m['n']})
| Metric | Score |
|--------|-------|
| Mean Brier | {m['mean_brier']:.4f} |
| Mean Baseline | {m['mean_baseline']:+.2f} |
"""
    
    report += """
## Score Interpretation
- **Baseline Score**: 0 = same as chance, >0 = better, <0 = worse
- **Peer Score**: 0 = same as community, >0 = better, <0 = worse
- **Skill Score**: >0% = better than naive baseline
"""
    
    return report


def generate_detailed_tables(grades: list[dict], run_data: dict = None) -> dict[str, str]:
    """
    Generate detailed text-based results tables for each question type.
    Returns a dict mapping filename (sluggified) to the table content string.
    """
    tables = {}
    
    # ============= BINARY =============
    binary_grades = [g for g in grades if g.get('question_type') == 'binary']
    if binary_grades:
        lines = []
        lines.append("=" * 130)
        lines.append("BINARY QUESTIONS RESULTS")
        lines.append("=" * 130 + "\n")
        lines.append(f"{'#':<3} {'Our Pred':<10} {'Comm Pred':<10} {'Outcome':<8} {'Log Score':<12} {'Peer Score':<12} Title")
        lines.append("-" * 130)
        
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
            lines.append(f"{i:<3} {our:<10.3f} {comm_str:<10} {outcome:<8} {log_score:<12.4f} {peer_str:<12} {title}{marker}")
        
        lines.append("-" * 130)
        lines.append(f"\nSUMMARY: {len(binary_grades)} binary questions")
        correct_count = sum(1 for g in binary_grades if (g.get('forecast', 0) > 0.5 and g.get('outcome') == 1) or (g.get('forecast', 0) < 0.5 and g.get('outcome') == 0))
        lines.append(f"Correct predictions: {correct_count}/{len(binary_grades)} ({100*correct_count/len(binary_grades):.1f}%)")
        peer_scores = [g['peer_score'] for g in binary_grades if g.get('peer_score') is not None]
        if peer_scores:
            lines.append(f"Mean Peer Score: {sum(peer_scores)/len(peer_scores):.2f} (n={len(peer_scores)}) (positive = beating community)")
        lines.append(f"Mean Log Score: {sum(g.get('log_score', 0) for g in binary_grades)/len(binary_grades):.4f}")
        
        tables["binary_results.txt"] = "\n".join(lines)

    # ============= MULTIPLE CHOICE =============
    mc_grades = [g for g in grades if g.get('question_type') == 'multiple_choice']
    if mc_grades:
        lines = []
        lines.append("=" * 140)
        lines.append("MULTIPLE CHOICE QUESTIONS RESULTS")
        lines.append("=" * 140 + "\n")
        
        for i, g in enumerate(mc_grades, 1):
            title = g.get('title', '')[:80]
            resolution = g.get('resolution', '')
            our_fc = g.get('forecast', {})
            comm_fc = g.get('community_forecast', {})
            our_prob = g.get('correct_probability', 0)
            
            # Recalculate comm_prob if needed (if it was a list)
            comm_prob = 0
            if isinstance(comm_fc, dict):
                comm_prob = comm_fc.get(resolution, 0)
            
            log_score = g.get('log_score', 0)
            brier = g.get('brier_score', 0)
            peer = g.get('peer_score')
            
            lines.append(f"\n[{i}] {title}")
            lines.append("-" * 110)
            lines.append(f"Resolution: {resolution}")
            comm_prob_str = f"{comm_prob:.3f}" if comm_prob else "N/A"
            line = f"Our prob on correct: {our_prob:.3f} | Community: {comm_prob_str}"
            if peer is not None:
                line += f" | Peer Score: {peer:.2f}"
            lines.append(line)
            lines.append(f"Log Score: {log_score:.4f} | Brier: {brier:.4f}\n")

            lines.append(f"{'Option':<55} {'Us':<10} {'Comm':<10} {'Correct?'}")
            lines.append("-" * 90)

            all_opts = sorted(set(our_fc.keys()) | set(comm_fc.keys() if isinstance(comm_fc, dict) else []))
            for opt in all_opts:
                our_p = our_fc.get(opt, 0)
                comm_p = comm_fc.get(opt) if isinstance(comm_fc, dict) else None
                comm_str = f"{comm_p:.3f}" if comm_p is not None else "N/A"
                is_correct = "  <-- CORRECT" if str(opt) == str(resolution) else ""
                lines.append(f"{str(opt)[:54]:<55} {our_p:<10.3f} {comm_str:<10} {is_correct}")
        
        lines.append("\n" + "=" * 140)
        lines.append(f"SUMMARY: {len(mc_grades)} multiple choice questions")
        peer_scores = [g['peer_score'] for g in mc_grades if g.get('peer_score') is not None]
        if peer_scores:
            lines.append(f"Mean Peer Score: {sum(peer_scores)/len(peer_scores):.2f} (n={len(peer_scores)})")
        lines.append(f"Mean Log Score: {sum(g.get('log_score', 0) for g in mc_grades)/len(mc_grades):.4f}")
        lines.append(f"Mean Brier Score: {sum(g.get('brier_score', 0) for g in mc_grades)/len(mc_grades):.4f}")
        
        tables["mc_results.txt"] = "\n".join(lines)

    # ============= NUMERIC =============
    num_grades = [g for g in grades if g.get('question_type') == 'numeric']
    if num_grades:
        lines = []
        lines.append("=" * 130)
        lines.append("NUMERIC QUESTIONS RESULTS (Normalized Density scoring)")
        lines.append("=" * 130 + "\n")
        lines.append(f"{'#':<3} {'Resolution':<14} {'Our NormD':<10} {'Comm NormD':<10} {'Our Log':<10} {'Peer':<10} Title")
        lines.append("-" * 130)

        for i, g in enumerate(num_grades, 1):
            res = g.get('resolution', 0)
            our_log = g.get('log_score', 0)
            peer = g.get('peer_score')
            
            # Use stored densities if available, else zero
            our_den = g.get('our_density_norm', 0)
            comm_den = g.get('community_density_norm', 0)
            
            # If not in the grade dict (older runs), we can't show them here easily
            # but for newly graded ones they will be there.
            
            peer_str = f"{peer:.2f}" if peer is not None else "N/A"
            title = g.get('title', '')[:65]

            res_str = f"{res:.4g}" if abs(res) > 1e6 else f"{res:.4f}"
            lines.append(f"{i:<3} {res_str:<14} {our_den:<10.3f} {comm_den:<10.3f} {our_log:<10.4f} {peer_str:<10} {title}")

        lines.append("-" * 130)
        lines.append(f"\nSUMMARY: {len(num_grades)} numeric questions")
        log_scores = [g.get('log_score') for g in num_grades if g.get('log_score') is not None]
        if log_scores:
            lines.append(f"Mean Log Score: {sum(log_scores)/len(log_scores):.4f}")
        
        peer_scores = [g['peer_score'] for g in num_grades if g.get('peer_score') is not None]
        if peer_scores:
            lines.append(f"Mean Peer Score: {sum(peer_scores)/len(peer_scores):.2f} (n={len(peer_scores)}) (positive = beating community)")
        
        tables["numeric_results.txt"] = "\n".join(lines)
        
    return tables
