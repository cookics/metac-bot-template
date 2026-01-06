# 🤖 Autonomous Backtesting Framework

This directory contains the framework for running end-to-end backtests, grading forecasts, and visualizing results. The entire process is now automated through a single entry point.

## 🚀 The Autonomous Workflow
To run a full backtest from scratch (Forecasting → Grading → Community Data Fetch → Report Generation), use the master script from the project root:

```bash
python backtesting/scripts/run_full_backtest.py --run-name backtest_6 --limit 50
```

### What this script does:
1.  **Forecasting**: Runs `backtest.py --run` to generate LLM forecasts using the current model.
2.  **Grading**: Runs `backtest.py --grade` to fetch resolutions and calculate initial scores.
3.  **Community Recovery**: Runs `fetch_fixed_community.py` to retrieve community forecasts via the CSV API for 100% coverage (including sub-questions).
4.  **Report Generation**: Runs `gen_tables.py` to produce final tables and high-res PDF visualizations.

---

## 📁 Directory Structure
All outputs are organized by run name under `data/runs/`:
- `data/runs/<name>/results/`: JSON grades and raw forecast data.
- `data/runs/<name>/plots/`: 
  - `binary_results.txt`: Detailed table with all 17/17 binary peer scores.
  - `mc_results.txt`: Detailed table for multiple-choice questions.
  - `numeric_results.txt`: Detailed table for numeric questions.
  - `pdf_*.png`: **High-resolution (200 DPI)** PDF density plots with Gaussian smoothing and community overlays.
  - `*_summary.png`: Combined visualization of all questions in the run.

---

## 📈 Visualizations: PDF Density Plots
We have moved from jagged CDF plots to smooth **Probability Density Function (PDF)** plots.
- **Smoothing**: Uses Gaussian filters to create clean, readable density curves.
- **Overlay**: Includes both "Our Forecast" (Blue) and "Community Forecast" (Green) for direct comparison.
- **Resolution**: Marked with a dashed red line and a specific density value at that point.

---

## 🔬 Scoring Logic: Normalized Density
For numeric questions, we use **Normalized Density** to calculate Log and Peer scores. This ensures that performance is comparable regardless of the question's scale (e.g., comparing a spread of 100 bps vs a price of $500).

### Formula
1. **Normalize**: Map the question range [min, max] to the unit interval [0, 1].
2. **Step Size**: $\Delta x_{norm} = 1 / 200$ (for our 201-point CDF).
3. **Density**: $f_{norm}(x) = \frac{CDF[i+1] - CDF[i]}{\Delta x_{norm}}$.
4. **Log Score**: $\ln(f_{norm}(x))$ (floored to $10^{-5}$).
5. **Peer Score**: $100 \times (\ln(Our\_f_{norm}) - \ln(Community\_f_{norm}))$.

---

## 🆘 Recovery: Community Data Fix ("The CSV Hell")
The standard Metaculus API often lacks historical community aggregations for **group sub-questions**.

### The Solution
We use a specialized retrieval process:
1. Identify the `post_id` and the specific `sub_question_id`.
2. Construct the download URL: `.../api/posts/{post_id}/download-data/?sub_question={sub_question_id}`.
3. Authenticate and download the ZIP file.
4. Extract `forecast_data.csv`.
5. Map the `Probability Yes` or `Continuous CDF` column from the `unweighted` or `recency_weighted` aggregation rows.

This recovery logic is handled automatically by the pipeline.
