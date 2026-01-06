
import json
import math

grades_path = 'backtest_results/run_20260105_192050_honest_50_fixed_dates.grades.json'
run_path = 'backtest_results/run_20260105_192050_honest_50_fixed_dates.json'

with open(grades_path) as f:
    grades = json.load(f)['grades']
with open(run_path) as f:
    forecasts = json.load(f)['forecasts']

title_to_forecast = {f['title']: f for f in forecasts}

for g in grades:
    if g.get('question_type') == 'numeric' and g.get('resolution') == -0.5562:
        print(f"MATCH FOUND: {g['title']}")
        print(f"Resolution: {g['resolution']}")
        
        comm_cdf = g.get('community_forecast')
        print(f"Comm CDF length: {len(comm_cdf) if comm_cdf else 'None'}")
        
        fc = title_to_forecast.get(g['title'], {})
        details = fc.get('question_details', {})
        scaling = details.get('scaling', {})
        range_min = scaling.get('range_min')
        range_max = scaling.get('range_max')
        print(f"Range: [{range_min}, {range_max}]")
        
        if comm_cdf and range_min is not None and range_max is not None:
            res = g['resolution']
            res_norm = (res - range_min) / (range_max - range_min)
            print(f"Res Norm: {res_norm}")
            
            idx_float = res_norm * 200
            idx_low = int(math.floor(idx_float))
            idx_high = int(math.ceil(idx_float))
            print(f"Indices: {idx_low}, {idx_high}")
            
            if 0 <= idx_low < 201 and 0 <= idx_high < 201:
                p_low = comm_cdf[idx_low]
                p_high = comm_cdf[idx_high]
                print(f"Probs: {p_low}, {p_high}")
                print(f"Delta P: {p_high - p_low}")
                
                delta_x = (idx_high - idx_low) * (range_max - range_min) / 200.0
                print(f"Delta X: {delta_x}")
                if delta_x > 0:
                    print(f"Density: {(p_high - p_low) / delta_x}")
            else:
                print("Indices out of range")
        break
