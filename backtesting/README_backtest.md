# Backtesting Infrastructure

This directory contains the framework for running backtests, grading forecasts, and visualizing results.

## Folder Structure
- `scripts/`: Python scripts for running backtests and processing data.
- `data/cache/`: Historical research/search data used to prevent API costs and ensure consistency.
- `data/results/`: JSON files containing raw forecast outputs and grade summaries.
- `data/plots/`: Generated results tables (`.txt`) and CDF visualizations (`.png`).

---

## 🔬 Scoring Logic: Normalized Density
For numeric questions, we use **Normalized Density** to calculate Log and Peer scores. This ensures that performance is comparable regardless of the question's scale (e.g., comparing a spread of 100 bps vs a price of \$500).

### Formula
1. **Normalize**: Map the question range $[min, max]$ to the unit interval $[0, 1]$.
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
2. Construct the download URL: `https://www.metaculus.com/api/posts/{post_id}/download-data/?sub_question={sub_question_id}`.
3. Authenticate and download the ZIP file.
4. Extract `forecast_data.csv`.
5. Map the `Probability Yes` or `Continuous CDF` column from the `unweighted` or `recency_weighted` aggregation rows.

This logic is encapsulated in `fetch_fixed_community.py`.

---

## How to Run (from Project Root)
1. **Rerun Backtest**: `python backtesting/scripts/backtest.py --run --config clean_run`
2. **Grade & Clean**: `python backtesting/scripts/backtest.py --grade --run-id latest`
3. **Fetch Community Data**: `python backtesting/scripts/fetch_fixed_community.py`
4. **Generate Reports**: `cd backtesting/scripts; python gen_tables.py; python regen_cdf.py`
