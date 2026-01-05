"""
Metaculus Forecasting Bot - Main Entry Point

This is the production orchestrator that:
1. Fetches open questions from Metaculus
2. Runs forecasting for each question
3. Submits predictions to Metaculus

For testing without submission, use test_forecast.py instead.
"""
import asyncio
import sys

# Ensure stdout handles Unicode correctly on Windows
if hasattr(sys, 'stdout') and sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from config import (
    SUBMIT_PREDICTION,
    USE_EXAMPLE_QUESTIONS,
    NUM_RUNS_PER_QUESTION,
    SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
    EXAMPLE_QUESTIONS,
)
from metaculus_api import (
    get_open_question_ids_from_tournament,
    get_post_details,
    post_question_prediction,
    post_question_comment,
    create_forecast_payload,
    forecast_is_already_made,
)
from forecasting import (
    get_binary_gpt_prediction,
    get_numeric_gpt_prediction,
    get_multiple_choice_gpt_prediction,
)


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    """
    Forecast a single question and optionally submit to Metaculus.
    """
    post_details = get_post_details(post_id)
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
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question
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

    if submit_prediction:
        forecast_payload = create_forecast_payload(forecast, question_type)
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    """
    Forecast all questions in the list.
    """
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
    
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
                f"Error: {forecast_summary.__class__.__name__} {forecast_summary}\n"
                f"URL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)


if __name__ == "__main__":
    print("Starting BOT")
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament()
    print(open_question_id_post_id)
    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
