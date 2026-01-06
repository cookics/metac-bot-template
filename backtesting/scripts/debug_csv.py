import requests
import zipfile
import io
import csv
from config import AUTH_HEADERS, API_BASE_URL

def debug_csv_structure(post_id, question_id):
    url = f"{API_BASE_URL}/posts/{post_id}/download-data/?sub_question={question_id}"
    print(f"Downloading CSV for Post {post_id}, Q {question_id}...")
    
    response = requests.get(url, **AUTH_HEADERS)
    if not response.ok:
        print(f"Failed: {response.status_code}")
        return
        
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = 'forecast_data.csv'
            if csv_name not in z.namelist():
                print(f"{csv_name} not found. Available: {z.namelist()}")
                return
            
            print(f"Inspecting {csv_name}...")
            with z.open(csv_name) as f:
                content = f.read().decode('utf-8')
                reader = csv.DictReader(io.StringIO(content))
                rows = list(reader)
                if not rows:
                    print("CSV is empty")
                    return
                print("Columns:", reader.fieldnames)
                # Print unique usernames
                usernames = set(r.get('Forecaster Username') for r in rows)
                print("Usernames present:", usernames)
                
                # Print last few rows
                for r in rows[-5:]:
                    print(f"Row: {r.get('Forecaster Username')} | ProbCategory: {r.get('Probability Yes Per Category')} | ProbYes: {r.get('Probability Yes')}")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # "Which states will Pope Leo XIV visit in 2025? (Turkiye)"
    debug_csv_structure(39723, 39102)
