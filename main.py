import asyncio
import datetime
import json
import os
import asyncio # Moved to top with other stdlib imports
# import datetime # No longer directly used in main.py
# import json # No longer directly used in main.py
# import os # No longer directly used in main.py
# import re # No longer directly used in main.py
# import dotenv # Removed, dotenv.load_dotenv() is now in config.py

# Third-party imports
# from openai import AsyncOpenAI # No longer directly used in main.py
# import numpy as np # No longer directly used in main.py
# import requests # No longer directly used in main.py

# Local application imports

# Import configurations from config.py
from config import (
    SUBMIT_PREDICTION,
    USE_EXAMPLE_QUESTIONS,
    NUM_RUNS_PER_QUESTION,
    SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
    GET_NEWS,
    OPENROUTER_API_KEY,
    METACULUS_TOKEN,
    OPENAI_API_KEY, # Used by call_llm_OAI and potentially by news.py's SmartSearcher
    TOURNAMENT_ID,
    EXAMPLE_QUESTIONS,
    AUTH_HEADERS,
    API_BASE_URL,
    CONCURRENT_REQUESTS_LIMIT # Used to initialize llm_rate_limiter
)

# The following constants are now defined in config.py:
# SUBMIT_PREDICTION, USE_EXAMPLE_QUESTIONS, NUM_RUNS_PER_QUESTION, SKIP_PREVIOUSLY_FORECASTED_QUESTIONS, GET_NEWS
# OPENROUTER_API_KEY, METACULUS_TOKEN, OPENAI_API_KEY
# Tournament IDs (Q4_2024_AI_BENCHMARKING_ID, etc.) and TOURNAMENT_ID
# EXAMPLE_QUESTIONS
# AUTH_HEADERS, API_BASE_URL
# CONCURRENT_REQUESTS_LIMIT (single definition now in config.py)

# region Comments
# Also, we realize the below code could probably be cleaned up a bit in a few places
# Though we are assuming most people will dissect it enough to make this not matter much
# Hopefully this is a good starting point for people to build on and get a gist of what's involved
# endregion

# llm_rate_limiter is now defined in llm_service.py

# Import question type specific prediction functions
from binary_question import get_binary_gpt_prediction
from numeric_question import get_numeric_gpt_prediction
from multiple_choice_question import get_multiple_choice_gpt_prediction

######################### HELPER FUNCTIONS #########################

# @title Helper functions
# AUTH_HEADERS and API_BASE_URL are now imported from config.py

# Import Metaculus API functions
from metaculus_api import (
    post_question_comment,
    post_question_prediction,
    create_forecast_payload,
    # list_posts_from_tournament, # Not directly used in main.py after refactor, but get_open_question_ids_from_tournament uses it
    get_open_question_ids_from_tournament,
    get_post_details
)

# from llm_service import call_llm_OAI, call_llm # Not directly used in main.py anymore

# Note: The main logic of main.py now calls the get_xxx_gpt_prediction functions from
# binary_question.py, numeric_question.py, and multiple_choice_question.py.
# These modules, in turn, call the necessary LLM functions from llm_service.py
# and research functions from news.py.

# The 'import config' line is removed as specific items are directly imported via 'from config import ...'

################### FORECASTING ###################
# This section (forecast_is_already_made, forecast_individual_question, forecast_questions)
# remains in main.py as it orchestrates the overall forecasting process.
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = get_post_details(post_id) # Imported from metaculus_api
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += f"-----------------------------------------------\nQuestion: {title}\n"
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details) # This function remains in main.py
        and config.SKIP_PREVIOUSLY_FORECASTED_QUESTIONS == True # Use config
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    # Call the appropriate question-specific prediction function
    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction( # Imported from binary_question.py
            question_details, config.NUM_RUNS_PER_QUESTION # Use config
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction( # Imported from numeric_question.py
            question_details, config.NUM_RUNS_PER_QUESTION # Use config
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_gpt_prediction( # Imported from multiple_choice_question.py
            question_details, config.NUM_RUNS_PER_QUESTION # Use config
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n")
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    if question_type == "numeric":
        summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{comment[:200]}...\n```\n\n"

    if config.SUBMIT_PREDICTION == True: # Use config.SUBMIT_PREDICTION
        forecast_payload = create_forecast_payload(forecast, question_type) # Imported from metaculus_api
        post_question_prediction(question_id, forecast_payload) # Imported from metaculus_api
        post_question_comment(post_id, comment) # Imported from metaculus_api
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    print("CHANGE")
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)




######################## FINAL RUN #########################
if __name__ == "__main__":
    print("Starting BOT")
    if config.USE_EXAMPLE_QUESTIONS: # Use config.USE_EXAMPLE_QUESTIONS
        open_question_id_post_id = config.EXAMPLE_QUESTIONS # Use config.EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament() # TOURNAMENT_ID is used as default arg
    print(open_question_id_post_id)
    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            config.SUBMIT_PREDICTION, # Use config.SUBMIT_PREDICTION
            config.NUM_RUNS_PER_QUESTION, # Use config.NUM_RUNS_PER_QUESTION
            config.SKIP_PREVIOUSLY_FORECASTED_QUESTIONS, # Use config.SKIP_PREVIOUSLY_FORECASTED_QUESTIONS
        )
    )
