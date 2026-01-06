import json
import os
from pathlib import Path

cache_dir = Path(r"c:\Users\cooki\Desktop\Spring Bot\metac-bot-template\backtest_cache")
removed_total = 0
files_processed = 0

for json_file in cache_dir.glob("*.json"):
    files_processed += 1
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'search_results' not in data:
        continue
        
    original_count = len(data['search_results'])
    # Strict filter: published_date must be present and not null
    cleaned_results = [
        res for res in data['search_results'] 
        if res.get('published_date') is not None
    ]
    
    removed_in_file = original_count - len(cleaned_results)
    if removed_in_file > 0:
        removed_total += removed_in_file
        data['search_results'] = cleaned_results
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Cleaned {json_file.name}: removed {removed_in_file} results with null date.")

print(f"\nCache cleaning complete.")
print(f"Files processed: {files_processed}")
print(f"Total results removed: {removed_total}")
