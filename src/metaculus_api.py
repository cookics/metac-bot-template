"""
Metaculus API interactions.
All functions here are standardized and should rarely need modification.
"""
import json
import requests
import zipfile
import io
import csv
from pathlib import Path
from config import AUTH_HEADERS, API_BASE_URL, TOURNAMENT_ID


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """
    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the API payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a 201 value cdf list.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str = TOURNAMENT_ID,
    offset: int = 0,
    count: int = 50,
    order_by: str = "-publish_time",
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": order_by,
        "forecast_type": ",".join(["binary", "multiple_choice", "numeric"]),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def get_open_question_ids_from_tournament(tournament_id: str = TOURNAMENT_ID) -> list[tuple[int, int, str]]:
    """
    Get all open question IDs from the configured tournament.
    Returns list of (question_id, post_id, title) tuples.
    """
    posts = list_posts_from_tournament(tournament_id=tournament_id)

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id, question["title"]))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(url, **AUTH_HEADERS)
    if not response.ok:
        raise Exception(response.text)
    details = json.loads(response.content)
    return details


def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts.
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


# ========================= BACKTESTING FUNCTIONS =========================

def get_resolved_questions_from_tournament(
    tournament_id: int | str = TOURNAMENT_ID,
    limit: int = 100,
) -> list[dict]:
    """
    Fetch resolved questions from a tournament for backtesting.
    
    Args:
        tournament_id: Tournament ID to fetch from
        limit: Maximum number of questions to fetch
    
    Returns:
        List of resolved question dicts with details
    """
    url = f"{API_BASE_URL}/posts/"
    
    all_results = []
    offset = 0
    
    while len(all_results) < limit:
        params = {
            "limit": min(50, limit - len(all_results)),
            "offset": offset,
            "order_by": "publish_time",  # Oldest first for even sampling
            "forecast_type": ",".join(["binary", "multiple_choice", "numeric"]),
            "tournaments": tournament_id,
            "statuses": "resolved",
            "include_description": "true",
        }
        
        response = requests.get(url, **AUTH_HEADERS, params=params)
        
        if not response.ok:
            print(f"API Error: {response.status_code} - {response.text[:200]}")
            break
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            break
        
        all_results.extend(results)
        offset += len(results)
        
        print(f"[Metaculus] Fetched {len(all_results)} resolved questions...")
    
    return all_results[:limit]


def sample_questions_evenly(questions: list[dict], n: int = 100) -> list[dict]:
    """
    Sample questions evenly by time (publish date).
    
    Args:
        questions: List of question dicts
        n: Number of questions to sample
    
    Returns:
        Evenly sampled subset
    """
    if len(questions) <= n:
        return questions
    
    # Sort by publish time using published_at field
    sorted_qs = sorted(questions, key=lambda q: q.get("published_at", ""))
    
    # Take evenly spaced samples
    step = len(sorted_qs) / n
    sampled = [sorted_qs[int(i * step)] for i in range(n)]
    
    return sampled


def get_question_resolution(question_details: dict) -> dict:
    """
    Extract resolution information from a question.
    
    Returns:
        Dict with resolution value and metadata
    """
    question = question_details.get("question", question_details)
    
    resolution_data = {
        "question_id": question.get("id"),
        "title": question.get("title"),
        "type": question.get("type"),
        "resolution": question.get("resolution"),
        "resolved_at": question.get("actual_resolve_time"),
        "created_at": question.get("created_at"),
    }
    
    # For binary questions
    if question.get("type") == "binary":
        res = question.get("resolution")
        resolution_data["resolved_yes"] = res == "yes" or res == True
        resolution_data["resolved_no"] = res == "no" or res == False
    
    return resolution_data


def extract_question_for_backtest(post: dict) -> dict:
    """
    Extract relevant fields from a post for backtesting.
    
    Returns:
        Simplified question dict for backtesting
    """
    question = post.get("question", {})
    
    return {
        "post_id": post.get("id"),
        "question_id": question.get("id"),
        "title": question.get("title"),
        "description": question.get("description"),
        "resolution_criteria": question.get("resolution_criteria"),
        "fine_print": question.get("fine_print"),
        "type": question.get("type"),
        "options": question.get("options"),  # For multiple choice
        "scaling": question.get("scaling"),  # For numeric
        "open_upper_bound": question.get("open_upper_bound"),
        "open_lower_bound": question.get("open_lower_bound"),
        "resolution": question.get("resolution"),
        "resolved_at": question.get("actual_resolve_time"),
        "created_at": question.get("created_at"),
        "publish_time": post.get("published_at"),
    }


def get_community_forecast(post_id: int, at_time: str = None) -> dict:
    """
    Get the community (aggregated) forecast for a question.
    
    Args:
        post_id: The post ID
        at_time: Optional ISO timestamp. If provided, returns the forecast active at this time.
                 If None, returns the latest/final forecast.
    
    Returns:
        Dict with community forecast data:
            - For binary: {"probability_yes": float}
            - For numeric: {"forecast_values": list[float]} (201-point CDF)
            - For multiple_choice: {"probability_yes_per_category": dict}
    """
    try:
        details = get_post_details(post_id)
        question = details.get("question", {})
        question_type = question.get("type")
        aggregations = question.get("aggregations", {})
        
        # Prefer recency_weighted for 'current' forecast, unweighted often has history
        agg_latest = aggregations.get("recency_weighted", {}).get("latest") or \
                     aggregations.get("unweighted", {}).get("latest")
        
        target_forecast = agg_latest
        
        # If at_time is provided, look through history
        if at_time and aggregations.get("unweighted", {}).get("history"):
            from datetime import datetime
            try:
                # Convert at_time to unix timestamp
                if isinstance(at_time, str):
                    target_ts = datetime.fromisoformat(at_time.replace("Z", "+00:00")).timestamp()
                else:
                    target_ts = float(at_time)
                
                history = aggregations.get("unweighted", {}).get("history", [])
                found = False
                for bin in history:
                    # Check if target_time falls within this bin
                    start = bin.get("start_time")
                    end = bin.get("end_time")
                    if start is not None and end is not None:
                        if start <= target_ts <= end:
                            target_forecast = bin
                            found = True
                            break
                
                if not found:
                    # Fallback logic:
                    # 1. If before history starts, use first available (initial forecast)
                    first_start = history[0].get("start_time") if history else None
                    last_end = history[-1].get("end_time") if history else None
                    
                    if history and first_start is not None and target_ts < first_start:
                         target_forecast = history[0]
                    # 2. If after history ends (e.g. question resolved but looking at late date), use last
                    elif history and last_end is not None and target_ts > last_end:
                         target_forecast = history[-1]
                    # 3. If mostly empty history but has items? Ensure we return something if possible.
                    elif history:
                         target_forecast = history[0]
                         
            except Exception as e:
                print(f"Error parsing time for community forecast: {e}")
        
        if not target_forecast:
            return {"error": "No community forecast available"}
            
        # Extract forecast data
        forecast_values = target_forecast.get("forecast_values")
        centers = target_forecast.get("centers")
        
        # If numeric and we have centers but no forecast_values (common in history), 
        # construct a CDF from the centers (samples)
        if question_type == "numeric" and not forecast_values and centers:
            try:
                scaling = question.get("scaling", {})
                r_min = scaling.get("range_min")
                r_max = scaling.get("range_max")
                
                if r_min is not None and r_max is not None:
                    # Normalize centers to 0-1 range
                    normalized_centers = []
                    range_size = r_max - r_min
                    if range_size > 0:
                        for c in centers:
                            norm = (c - r_min) / range_size
                            normalized_centers.append(max(0.0, min(1.0, norm)))
                    
                    if normalized_centers:
                        normalized_centers.sort()
                        # Create 201-point CDF
                        constructed_cdf = []
                        n_samples = len(normalized_centers)
                        
                        for i in range(201):
                            x = i / 200.0
                            # Count fraction of samples <= x
                            # Using simple empirical CDF
                            count = sum(1 for s in normalized_centers if s <= x)
                            prob = count / n_samples
                            constructed_cdf.append(prob)
                        
                        forecast_values = constructed_cdf
            except Exception as e:
                print(f"Error constructing CDF from centers: {e}")
        
        # If multiple choice and we have centers (probabilities) but no options mapping, map them
        if question_type == "multiple_choice":
            # Start with probability_yes_per_category if available (latest forecast)
            probs = target_forecast.get("probability_yes_per_category")
            
            # If not available (history often just has centers), map centers to options
            if not probs and centers:
                options = question.get("options", [])
                if options and len(options) == len(centers):
                    probs = dict(zip(options, centers))
            
            # Add to result for grading
            if probs:
                return {
                    "question_type": "multiple_choice",
                    "probability_yes_per_category": probs,
                    "forecaster_count": target_forecast.get("forecaster_count", 0),
                }

        return {
            "question_type": question_type,
            "forecast_values": forecast_values,  # For numeric (CDF)
            "centers": centers, 
            "probability_yes": target_forecast.get("centers", [None])[0] if question_type == "binary" else None,
            "forecaster_count": target_forecast.get("forecaster_count", 0),
        }
    
    except Exception as e:
        return {"error": str(e)}


def get_community_forecast_from_csv(post_id: int, question_id: int) -> dict:
    """
    Fetch community forecast by downloading the ZIP/CSV data from Metaculus.
    Works for both standalone questions and group sub-questions.
    
    Tries two approaches:
    1. Download without sub_question parameter (standalone questions)
    2. Download with sub_question parameter (group sub-questions)
    """
    # Check cache first
    cache_dir = Path(__file__).resolve().parent.parent / "backtesting" / "data" / "cache" / "community"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"comm_{post_id}_{question_id}.json"
    
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cached = json.load(f)
            # Don't return cached errors - retry instead
            if "error" not in cached:
                return cached
    
    def try_download(url):
        """Try to download and parse CSV from given URL."""
        try:
            response = requests.get(url, **AUTH_HEADERS)
            if not response.ok:
                return {"error": f"Download failed: {response.status_code}"}
                
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = 'forecast_data.csv'
                if csv_name not in z.namelist():
                    csv_files = [n for n in z.namelist() if n.endswith('.csv')]
                    if not csv_files: return {"error": "No CSV in ZIP"}
                    csv_name = csv_files[0]
                    
                with z.open(csv_name) as f:
                    content = f.read().decode('utf-8')
                    reader = csv.DictReader(io.StringIO(content))
                    rows = list(reader)
                    
                    # Filter for aggregation rows
                    agg_rows = [r for r in rows if r.get('Forecaster Username') in ['unweighted', 'recency_weighted']]
                    if not agg_rows:
                        return {"error": "No aggregation rows found in CSV"}
                    
                    # Take the latest aggregation row
                    target = agg_rows[-1]
                    result = {"source": "csv"}
                    
                    # 1. Binary
                    if target.get('Probability Yes'):
                        try:
                            result["question_type"] = "binary"
                            result["probability_yes"] = float(target.get('Probability Yes'))
                        except: pass
                    
                    # 2. Multiple Choice
                    if not result.get("probability_yes") and target.get('Probability Yes Per Category'):
                        try:
                            val_str = target.get('Probability Yes Per Category').replace("'", '"')
                            probs = json.loads(val_str)
                            if 'yes' in probs and len(probs) == 1:
                                result["question_type"] = "binary"
                                result["probability_yes"] = probs['yes']
                            else:
                                result["question_type"] = "multiple_choice"
                                result["probability_yes_per_category"] = probs
                        except: pass
                            
                    # 3. Numeric (CDF)
                    if target.get('Continuous CDF'):
                        try:
                            val_str = target.get('Continuous CDF')
                            cdf = json.loads(val_str)
                            if isinstance(cdf, list) and len(cdf) == 201:
                                # Reject dummy CDFs (all 1.0 or all 0.0)
                                if all(v == 1.0 for v in cdf) or all(v == 0.0 for v in cdf):
                                    return {"error": "CSV contains dummy all-ones or all-zeros CDF"}
                                result["question_type"] = "numeric"
                                result["forecast_values"] = cdf
                        except: pass
                    
                    # Try 'Probability' for binary if 'Probability Yes' is missing
                    if "question_type" not in result and target.get('Probability'):
                        try:
                            result["question_type"] = "binary"
                            result["probability_yes"] = float(target.get('Probability'))
                        except: pass
                    
                    if "question_type" not in result:
                        return {"error": "Could not extract forecast from CSV (missing valid data columns)"}
                    
                    return result
                    
        except Exception as e:
            return {"error": f"CSV processing error: {str(e)}"}
    
    # Try 1: Download without sub_question parameter (for standalone questions)
    url_standalone = f"{API_BASE_URL}/posts/{post_id}/download-data/"
    print(f"  [Metaculus] Trying CSV download for Post {post_id} (standalone)...")
    result = try_download(url_standalone)
    
    if "error" not in result:
        # Success! Save to cache and return
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)
        return result
    
    # Try 2: Download with sub_question parameter (for group sub-questions)
    url_subq = f"{API_BASE_URL}/posts/{post_id}/download-data/?sub_question={question_id}"
    print(f"  [Metaculus] Trying CSV download for Q{question_id} in Post {post_id} (sub_question)...")
    result = try_download(url_subq)
    
    if "error" not in result:
        # Success! Save to cache and return
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)
        return result
    
    # Both failed - return the last error (but don't cache it)
    return result
