import requests
import json
from config import API_BASE_URL, AUTH_HEADERS, TOURNAMENT_ID

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
        raise RuntimeError(f"Failed to post comment: {response.status_code} {response.text}")

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
        raise RuntimeError(f"Failed to post prediction: {response.status_code} {response.text}")

def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
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
    # For numeric questions, the 'forecast' argument is expected to be the continuous_cdf itself.
    if question_type == "numeric": # Could also be "date" - they use same payload structure
        return {
            "probability_yes": None,
            "probability_yes_per_category": None,
            "continuous_cdf": forecast, # Assign the forecast (CDF list) here
        }
    else: # Should not happen if question_type is always binary, multiple_choice, or numeric/date
        # Fallback or raise error for unknown type, though current structure implies this handles numeric/date
        # For safety, and to match previous structure if "date" was distinct but handled by this else:
        # This explicit check for "numeric" is better. If "date" questions need specific handling,
        # it would need its own block or be confirmed to use the numeric structure.
        # Given the original comment, this block was for numeric/date.
        # Assuming 'forecast' for a 'date' question would also be a CDF.
        return {
            "probability_yes": None,
            "probability_yes_per_category": None,
            "continuous_cdf": forecast, # Correctly assign forecast to continuous_cdf
        }
        # Removing the original comment as the issue is addressed.

def list_posts_from_tournament(
    tournament_id: int = TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    if not response.ok:
        raise Exception(f"Failed to list posts: {response.status_code} {response.text}")
    data = json.loads(response.content)
    return data

def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    """
    Retrieves open question IDs and their corresponding post IDs from a tournament.
    """
    posts_data = list_posts_from_tournament() # TOURNAMENT_ID from config is used by default

    post_dict = dict()
    for post in posts_data.get("results", []): # Added .get for safety
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
                open_question_id_post_id.append((question["id"], post_id))
    return open_question_id_post_id

def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(url, **AUTH_HEADERS)
    if not response.ok:
        raise Exception(f"Failed to get post details: {response.status_code} {response.text}")
    details = json.loads(response.content)
    return details
