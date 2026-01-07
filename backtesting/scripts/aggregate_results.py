import os
import json
import csv
from pathlib import Path

def aggregate_results():
    data_dir = Path("backtesting/data/runs")
    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store all data: {question_id: {title, type, resolution, models: {model_key: {forecast, peer_score, brier_score, crps}}}}
    aggregated_data = {}
    all_models = set()

    # Find all .grades.json files in backtest_7 through backtest_10
    for i in range(7, 11):
        backtest_path = data_dir / f"backtest_{i}" / "results"
        if not backtest_path.exists():
            continue
        
        for file_path in backtest_path.glob("*.grades.json"):
            # Use filename without timestamp as model key if possible, or just the whole name
            # Format: run_YYYYMMDD_HHMMSS_MODEL_NAME.grades.json
            parts = file_path.name.replace(".grades.json", "").split("_")
            if len(parts) >= 4:
                model_name = parts[3]
            else:
                model_name = file_path.stem.replace(".grades", "")
            
            # If multiple runs for the same model in different backtests, they might overwrite
            # For now, let's include the backtest ID in the model key to distinguish
            model_key = f"{model_name} (BT{i})"
            all_models.add(model_key)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                grades = data.get("grades", [])
                
                for grade in grades:
                    q_id = str(grade.get("question_id"))
                    if q_id not in aggregated_data:
                        aggregated_data[q_id] = {
                            "title": grade.get("title"),
                            "question_type": grade.get("question_type"),
                            "resolution": grade.get("resolution") if grade.get("question_type") != "binary" else grade.get("outcome"),
                            "models": {}
                        }
                    
                    aggregated_data[q_id]["models"][model_key] = {
                        "forecast": grade.get("forecast"),
                        "peer_score": grade.get("peer_score"),
                        "brier_score": grade.get("brier_score"),
                        "skill_score": grade.get("skill_score")
                    }

    # Sort models by name
    sorted_models = sorted(list(all_models))

    # Helper for formatting forecasts
    def format_forecast(forecast, q_type, resolution=None):
        if q_type == "binary":
            return f"{float(forecast)*100:.1f}%"
        elif q_type == "multiple_choice":
            if isinstance(forecast, dict) and resolution:
                # Find the probability for the correct resolution
                # Some resolutions might have extra whitespace or different casing
                res_key = str(resolution).strip()
                if res_key in forecast:
                    return f"{forecast[res_key]*100:.1f}%"
                # Fallback to case-insensitive match
                for k, v in forecast.items():
                    if k.strip().lower() == res_key.lower():
                        return f"{v*100:.1f}%"
                return "N/A"
            elif isinstance(forecast, dict):
                return "\n".join([f"{k}: {v*100:.1f}%" for k, v in forecast.items()])
            return str(forecast)
        elif q_type == "numeric":
            return "CDF"
        return str(forecast)

    # Split and write CSVs
    for q_type in ["binary", "multiple_choice", "numeric"]:
        rows = []
        # Header
        header = ["Question ID", "Title", "Resolution"]
        for model in sorted_models:
            header.append(f"{model} Forecast")
        for model in sorted_models:
            header.append(f"{model} Peer Score")
        rows.append(header)

        peer_scores_by_model = {model: [] for model in sorted_models}

        for q_id, q_data in aggregated_data.items():
            if q_data["question_type"] != q_type:
                continue
            
            row = [q_id, q_data["title"], q_data["resolution"]]
            # Add all forecasts first
            for model in sorted_models:
                m_data = q_data["models"].get(model, {})
                forecast_val = format_forecast(m_data.get("forecast"), q_type, q_data["resolution"]) if "forecast" in m_data else "N/A"
                row.append(forecast_val)
            
            # Add all peer scores after
            for model in sorted_models:
                m_data = q_data["models"].get(model, {})
                peer_val = m_data.get('peer_score', 'N/A')
                row.append(f"{peer_val:.2f}" if isinstance(peer_val, (int, float)) else "N/A")
                
                if isinstance(peer_val, (int, float)):
                    peer_scores_by_model[model].append(peer_val)
            rows.append(row)

        # Average row
        avg_row = ["AVERAGE", "", ""]
        # Empty cells for forecasts
        for _ in sorted_models:
            avg_row.append("")
        # Averages for peer scores
        for model in sorted_models:
            scores = peer_scores_by_model[model]
            avg = sum(scores) / len(scores) if scores else "N/A"
            avg_row.append(f"{avg:.2f}" if isinstance(avg, (int, float)) else "N/A")
        rows.append(avg_row)

        csv_path = output_dir / f"aggregate_{q_type}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Wrote {csv_path}")

    # Write Markdown Summary (keeping for easy viewing, though user prefers CSV)
    md_path = output_dir / "aggregate_summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Consolidated Backtest Results\n\n")
        
        for q_type in ["binary", "multiple_choice", "numeric"]:
            f.write(f"## {q_type.replace('_', ' ').title()} Results\n\n")
            
            # Filter questions for this type
            type_questions = {k: v for k, v in aggregated_data.items() if v["question_type"] == q_type}
            if not type_questions:
                f.write("No questions found.\n\n")
                continue

            # Identify models that have data for this type
            models_with_data = []
            for model in sorted_models:
                if any(model in q["models"] for q in type_questions.values()):
                    models_with_data.append(model)
            
            # Markdown Table Header
            header = "| QID | Title | Resolution | " + " | ".join([f"{m} (Frc / Peer)" for m in models_with_data]) + " |"
            sep = "| --- | --- | --- | " + " | ".join(["---" for _ in models_with_data]) + " |"
            f.write(header + "\n")
            f.write(sep + "\n")

            for q_id, q_data in type_questions.items():
                title_short = (q_data["title"][:50] + "...") if len(q_data["title"]) > 50 else q_data["title"]
                res = q_data["resolution"]
                row = [q_id, title_short, str(res)]
                for model in models_with_data:
                    m_data = q_data["models"].get(model, {})
                    frc = format_forecast(m_data.get("forecast"), q_type, q_data["resolution"]) if "forecast" in m_data else "N/A"
                    peer = f"{m_data.get('peer_score', 0):.2f}" if isinstance(m_data.get('peer_score'), (int, float)) else "N/A"
                    row.append(f"{frc} / {peer}")
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

    # NEW: Consolidated Aggregate Summary CSV
    all_rows = []
    all_header = ["Question ID", "Type", "Title", "Resolution"]
    for model in sorted_models:
        all_header.append(f"{model} Forecast")
    for model in sorted_models:
        all_header.append(f"{model} Peer Score")
    all_rows.append(all_header)

    peer_scores_by_model_all = {model: [] for model in sorted_models}

    for q_id, q_data in aggregated_data.items():
        q_type = q_data["question_type"]
        row = [q_id, q_type, q_data["title"], q_data["resolution"]]
        
        # Add all forecasts first
        for model in sorted_models:
            m_data = q_data["models"].get(model, {})
            if "forecast" in m_data:
                frc = format_forecast(m_data.get("forecast"), q_type, q_data["resolution"])
                row.append(frc)
            else:
                row.append("N/A")
        
        # Add all peer scores after
        for model in sorted_models:
            m_data = q_data["models"].get(model, {})
            peer = m_data.get('peer_score', 'N/A')
            row.append(f"{peer:.2f}" if isinstance(peer, (int, float)) else "N/A")
            if isinstance(peer, (int, float)):
                peer_scores_by_model_all[model].append(peer)
        
        all_rows.append(row)

    # Average row for all_csv
    all_avg_row = ["AVERAGE", "", "", ""]
    # Empty cells for forecasts
    for _ in sorted_models:
        all_avg_row.append("")
    # Averages for peer scores
    for model in sorted_models:
        scores = peer_scores_by_model_all[model]
        avg = sum(scores) / len(scores) if scores else "N/A"
        all_avg_row.append(f"{avg:.2f}" if isinstance(avg, (int, float)) else "N/A")
    all_rows.append(all_avg_row)

    all_csv_path = output_dir / "aggregate_all.csv"
    with open(all_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
    print(f"Wrote {all_csv_path}")

    print(f"Wrote {md_path}")

if __name__ == "__main__":
    aggregate_results()
