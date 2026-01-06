"""
Visualization functions for backtesting results.

Generates:
- CDF plots for continuous questions
- Score comparison charts
- Summary statistics
"""
import json
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, visualization disabled")


PLOTS_DIR = Path(__file__).resolve().parent.parent / "data" / "plots"


def ensure_plots_dir():
    PLOTS_DIR.mkdir(exist_ok=True)


def cdf_to_pdf(cdf: list[float], smooth: bool = True) -> list[float]:
    """
    Convert a 201-point CDF to PDF by taking finite differences.
    Optionally applies Gaussian smoothing for a cleaner plot.
    """
    if not cdf or len(cdf) != 201:
        return [0.0] * 201
    
    # Take differences (PDF is derivative of CDF)
    pdf = [0.0] * 201
    for i in range(200):
        pdf[i] = (cdf[i+1] - cdf[i]) * 200  # Scale by 200 to normalize
    pdf[200] = pdf[199]  # Copy last value
    
    # Gaussian smoothing for less jagged appearance
    if smooth:
        try:
            import numpy as np
            from scipy.ndimage import gaussian_filter1d
            pdf = gaussian_filter1d(np.array(pdf), sigma=3).tolist()
        except ImportError:
            # Manual simple moving average as fallback
            window = 5
            smoothed = []
            for i in range(len(pdf)):
                start = max(0, i - window//2)
                end = min(len(pdf), i + window//2 + 1)
                smoothed.append(sum(pdf[start:end]) / (end - start))
            pdf = smoothed
    
    return pdf


def plot_pdf(
    cdf: list[float],
    resolution: float,
    range_min: float,
    range_max: float,
    title: str = "Probability Density",
    save_path: Optional[Path] = None,
    community_cdf: list[float] = None
) -> Optional[Path]:
    """
    Plot a PDF (probability density) with the resolution point marked.
    Converts the 201-point CDF to a smooth density curve.
    
    Args:
        cdf: 201-point CDF
        resolution: Actual resolved value
        range_min: Question range minimum
        range_max: Question range maximum
        title: Plot title
        save_path: Where to save the plot
        community_cdf: Optional community CDF for comparison
    
    Returns:
        Path to saved plot, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        return None
    
    ensure_plots_dir()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Convert CDF to PDF
    our_pdf = cdf_to_pdf(cdf, smooth=True)
    
    # Interpolate to 1000 points for smoother plotting
    import numpy as np
    x_201 = np.linspace(range_min, range_max, 201)
    x_1000 = np.linspace(range_min, range_max, 1000)
    our_pdf_interp = np.interp(x_1000, x_201, our_pdf)
    
    # Plot our density as filled curve
    ax.fill_between(x_1000, our_pdf_interp, alpha=0.3, color='blue', label='_nolegend_')
    ax.plot(x_1000, our_pdf_interp, 'b-', linewidth=2.5, label='Our Forecast')
    
    # Plot community density if available
    if community_cdf and len(community_cdf) == 201:
        comm_pdf = cdf_to_pdf(community_cdf, smooth=True)
        comm_pdf_interp = np.interp(x_1000, x_201, comm_pdf)
        ax.fill_between(x_1000, comm_pdf_interp, alpha=0.2, color='green', label='_nolegend_')
        ax.plot(x_1000, comm_pdf_interp, 'g-', linewidth=2.5, alpha=0.8, label='Community')
    
    # Mark the resolution with a vertical line
    ax.axvline(x=resolution, color='red', linestyle='--', linewidth=2.5, label=f'Resolution: {resolution:.4g}')
    
    # Find density at resolution point and mark it
    res_idx = min(200, max(0, int((resolution - range_min) / (range_max - range_min) * 200)))
    our_density_at_res = our_pdf[res_idx]
    ax.scatter([resolution], [our_density_at_res], color='red', s=150, zorder=5, edgecolors='black', linewidths=1)
    ax.annotate(f'Density = {our_density_at_res:.3f}', 
                xy=(resolution, our_density_at_res),
                xytext=(15, 10), textcoords='offset points',
                fontsize=11, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Save at higher resolution
    if save_path is None:
        save_path = PLOTS_DIR / "pdf_plot.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)  # Higher DPI for cleaner output
    plt.close()
    
    return save_path


# Keep old name as alias for compatibility
def plot_cdf(
    cdf: list[float],
    resolution: float,
    range_min: float,
    range_max: float,
    title: str = "CDF Visualization",
    save_path: Optional[Path] = None,
    community_cdf: list[float] = None
) -> Optional[Path]:
    """Alias for plot_pdf - now plots density instead of CDF."""
    return plot_pdf(cdf, resolution, range_min, range_max, title, save_path, community_cdf)


def plot_score_comparison(grades: list[dict], save_path: Optional[Path] = None) -> Optional[Path]:
    """
    Plot our scores vs baseline/community.
    
    Args:
        grades: List of graded forecasts
        save_path: Where to save
    
    Returns:
        Path to saved plot
    """
    if not HAS_MATPLOTLIB:
        return None
    
    ensure_plots_dir()
    
    # Extract scores
    baseline_scores = [g.get("baseline_score", 0) for g in grades if "baseline_score" in g]
    peer_scores = [g.get("peer_score") for g in grades if "peer_score" in g]
    peer_scores = [p for p in peer_scores if p is not None]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline scores histogram
    if baseline_scores:
        axes[0].hist(baseline_scores, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', label='Baseline (chance)')
        axes[0].set_xlabel('Baseline Score')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Baseline Scores (n={len(baseline_scores)}, mean={sum(baseline_scores)/len(baseline_scores):.1f})')
        axes[0].legend()
    
    # Peer scores histogram
    if peer_scores:
        axes[1].hist(peer_scores, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', label='Community median')
        axes[1].set_xlabel('Peer Score')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Peer Scores (n={len(peer_scores)}, mean={sum(peer_scores)/len(peer_scores):.1f})')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No peer scores available', ha='center', va='center')
        axes[1].set_title('Peer Scores')
    
    if save_path is None:
        save_path = PLOTS_DIR / "score_comparison.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    return save_path


def plot_categorical_summary(grades: list[dict], save_path: Optional[Path] = None) -> Optional[Path]:
    """Big combined plot for Binary/MC questions showing Outcome vs Community vs Our Prediction."""
    if not HAS_MATPLOTLIB:
        return None
    
    cat_grades = [g for g in grades if g.get("question_type") in ["binary", "multiple_choice"]]
    if not cat_grades:
        return None
        
    ensure_plots_dir()
    
    # Sort by title for stable ordering
    cat_grades.sort(key=lambda x: x.get("title", ""))
    
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(cat_grades) + 2))
    
    y_labels = []
    our_probs = []
    comm_probs = []
    outcomes = []
    
    for g in cat_grades:
        title = g.get("title", "Question")[:50]
        q_type = g.get("question_type")
        res = g.get("resolution")
        
        y_labels.append(f"{title} ({q_type})")
        
        if q_type == "binary":
            # Binary grades store outcome as int 0/1, not resolution string
            res_val = float(g.get("outcome", 0))
            our_val = g.get("forecast", 0.5)
            # community_forecast for binary is the float probability
            comm_val = g.get("community_forecast", 0.5) if g.get("community_forecast") is not None else 0.5
        else:
            # For MC, simplified view: correct category probability
            # g['forecast'] is a dict
            our_val = g.get("forecast", {}).get(res, 0.0) if isinstance(g.get("forecast"), dict) else 0.0
            comm_val = g.get("community_forecast", {}).get(res, 0.0) if isinstance(g.get("community_forecast"), dict) else 0.0
            res_val = 1.0 # 100% since it's the correct category
            
        our_probs.append(our_val)
        comm_probs.append(comm_val)
        outcomes.append(res_val)
        
    y_pos = range(len(y_labels))
    
    # Plot bars
    ax.barh([y - 0.2 for y in y_pos], our_probs, height=0.3, label='Our Forecast', color='blue', alpha=0.7)
    ax.barh([y + 0.1 for y in y_pos], comm_probs, height=0.3, label='Community', color='green', alpha=0.7)
    
    # Plot outcomes as markers
    ax.scatter(outcomes, y_pos, color='red', marker='|', s=500, label='Outcome', zorder=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Probability of Correct Outcome')
    ax.set_title('Categorical Questions: Us vs Community vs Resolution')
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)
    
    if save_path is None:
        save_path = PLOTS_DIR / "categorical_summary.png"
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


def plot_numeric_summary(grades: list[dict], save_path: Optional[Path] = None) -> Optional[Path]:
    """Combined plot for Numeric questions showing Resolution vs Community vs Our Forecast."""
    if not HAS_MATPLOTLIB:
        return None
        
    num_grades = [g for g in grades if g.get("question_type") == "numeric"]
    if not num_grades:
        return None
        
    ensure_plots_dir()
    num_grades.sort(key=lambda x: x.get("title", ""))
    
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(num_grades) + 2))
    
    y_labels = []
    
    for i, g in enumerate(num_grades):
        title = g.get("title", "Question")[:50]
        res = float(g.get("resolution", 0))
        y_labels.append(title)
        
        # Scaling for this question
        # Note: in numeric summary, normalized 0-1 range is easier to compare across questions
        # We use the resolution_idx/201 as normalized pos
        our_cdf = g.get("forecast", [0.5]*201)
        comm_cdf = g.get("community_forecast")
        res_idx = g.get("resolution_idx", 100)
        
        # Plot our distribution as a density gradient or box? 
        # For simplicity: plot median (50th percentile) and 25-75 range
        def get_percentile(cdf, p):
            for idx, val in enumerate(cdf):
                if val >= p: return idx / 200.0
            return 1.0
            
        our_med = get_percentile(our_cdf, 0.5)
        our_low = get_percentile(our_cdf, 0.25)
        our_high = get_percentile(our_cdf, 0.75)
        
        # Our span
        ax.plot([our_low, our_high], [i-0.2, i-0.2], 'b-', alpha=0.5, linewidth=4)
        ax.plot(our_med, i-0.2, 'bo', markersize=8, label='Our Median' if i==0 else "")
        
        # Community span
        if comm_cdf and len(comm_cdf) == 201:
            comm_med = get_percentile(comm_cdf, 0.5)
            comm_low = get_percentile(comm_cdf, 0.25)
            comm_high = get_percentile(comm_cdf, 0.75)
            ax.plot([comm_low, comm_high], [i+0.1, i+0.1], 'g-', alpha=0.5, linewidth=4)
            ax.plot(comm_med, i+0.1, 'go', markersize=8, label='Comm Median' if i==0 else "")
            
        # Resolution
        ax.plot(res_idx / 200.0, i, 'rx', markersize=10, mew=2, label='Resolution' if i==0 else "")
        
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Normalized Range (0 = Min, 1 = Max)')
    ax.set_title('Numeric Questions: Our Range vs Community vs Resolution')
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)
    
    if save_path is None:
        save_path = PLOTS_DIR / "numeric_summary.png"
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


def generate_all_plots(grades: list[dict], forecasts: list[dict] = None, plots_dir: Path = PLOTS_DIR) -> list[Path]:
    """
    Generate all visualization plots for a backtest run.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualizations")
        return []
    
    plots = []
    
    # 1. Individual Question Plots (Summary)
    cat_summary = plot_categorical_summary(grades, save_path=plots_dir / "categorical_summary.png")
    if cat_summary:
        plots.append(cat_summary)
        
    num_summary = plot_numeric_summary(grades, save_path=plots_dir / "numeric_summary.png")
    if num_summary:
        plots.append(num_summary)

    # 2. Score comparison stats
    score_plot = plot_score_comparison(grades, save_path=plots_dir / "score_comparison.png")
    if score_plot:
        plots.append(score_plot)
    
    # 3. Individual CDF plots for numeric questions (first 10)
    if forecasts:
        numeric_forecasts = [f for f in forecasts if f.get("question_type") == "numeric" and "forecast" in f][:10]
        
        for i, fc in enumerate(numeric_forecasts):
            cdf = fc.get("forecast", [])
            if not cdf or len(cdf) != 201:
                continue
            
            details = fc.get("question_details", {})
            scaling = details.get("scaling", {})
            
            # Find community forecast in grades if exists
            qid = fc.get("question_id")
            title = fc.get("title")
            grade_entry = next((g for g in grades if g.get("question_id") == qid), {})
            
            # Fallback to title if QID didn't match (for older grades files)
            if not grade_entry:
                grade_entry = next((g for g in grades if g.get("title") == title), {})
                
            comm_cdf = grade_entry.get("community_forecast")
            
            plot_path = plots_dir / f"cdf_{i+1}_{qid}.png"
            
            result = plot_cdf(
                cdf=cdf,
                resolution=float(fc.get("resolution", 0)) if fc.get("resolution") not in [None, "annulled"] else 0,
                range_min=scaling.get("range_min", 0),
                range_max=scaling.get("range_max", 100),
                title=fc.get("title", "Numeric Question")[:60] + "...",
                save_path=plot_path,
                community_cdf=comm_cdf
            )
            
            if result:
                plots.append(result)
    
    return plots
