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

# Add root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Minimum PDF value (like Metaculus's 0.01 floor)
MIN_PDF = 0.01


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
    
    result = {
        "question_type": "numeric",
        "resolution": resolution,
        "resolution_idx": resolution_idx,
        "predicted_percentile": cdf[resolution_idx],
        "crps": forecast_crps,
        "log_score": log_score_continuous(cdf, resolution_idx),
        "baseline_score": baseline_score_continuous(cdf, resolution_idx),
        "skill_score": skill_score,
        "brier_score": forecast_crps,  # For aggregate compatibility
    }
    
    if community_cdf and len(community_cdf) == 201:
        result["community_forecast"] = community_cdf
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
        community = community_forecast if isinstance(community_forecast, dict) else None
        # Unwrap if nested
        if community and "probability_yes_per_category" in community:
            community = community["probability_yes_per_category"]
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
