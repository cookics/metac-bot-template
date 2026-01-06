"""
Test Forecasting Script

This script runs the forecasting pipeline WITHOUT submitting to Metaculus.
Use this to:
- Test your prompts and forecasting logic
- Debug forecast extraction
- Preview what would be submitted before going live

Usage:
    python test_forecast.py                    # Test with example questions
    python test_forecast.py --post-id 12345   # Test specific question
"""
import asyncio
import argparse

from config import EXAMPLE_QUESTIONS, NUM_RUNS_PER_QUESTION
from metaculus_api import get_post_details
from forecasting import (
    get_binary_gpt_prediction,
    get_numeric_gpt_prediction,
    get_multiple_choice_gpt_prediction,
)


async def test_forecast_question(post_id: int, num_runs: int = 1) -> None:
    """
    Run a forecast for a single question and display results.
    Does NOT submit to Metaculus.
    """
    print(f"\n{'='*60}")
    print(f"TESTING FORECAST - Post ID: {post_id}")
    print(f"{'='*60}\n")

    # Fetch question details
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    
    title = question_details["title"]
    question_type = question_details["type"]
    question_id = question_details["id"]

    print(f"Title: {title}")
    print(f"Type: {question_type}")
    print(f"URL: https://www.metaculus.com/questions/{post_id}/")
    print(f"Description: {question_details.get('description', 'N/A')[:200]}...")
    print(f"\n{'-'*60}\n")

    # Generate forecast based on question type
    try:
        if question_type == "binary":
            forecast, comment = await get_binary_gpt_prediction(
                question_details, num_runs
            )
            print(f"FORECAST RESULT: {forecast * 100:.1f}% probability of Yes")
            
        elif question_type == "numeric":
            forecast, comment = await get_numeric_gpt_prediction(
                question_details, num_runs
            )
            print(f"FORECAST RESULT (CDF, first 10 values): {forecast[:10]}...")
            
        elif question_type == "multiple_choice":
            options = question_details["options"]
            print(f"Options: {options}")
            forecast, comment = await get_multiple_choice_gpt_prediction(
                question_details, num_runs
            )
            print(f"FORECAST RESULT:")
            for option, prob in forecast.items():
                print(f"  {option}: {prob * 100:.1f}%")
        else:
            print(f"Unknown question type: {question_type}")
            return

        print(f"\n{'-'*60}")
        print("FULL RATIONALE:")
        print(f"{'-'*60}")
        print(comment)
        
    except Exception as e:
        print(f"ERROR during forecasting: {e}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="Test forecasting without submitting to Metaculus")
    parser.add_argument(
        "--post-id", 
        type=int, 
        help="Specific post ID to test. If not provided, uses example questions."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS_PER_QUESTION,
        help=f"Number of runs per question (default: {NUM_RUNS_PER_QUESTION})"
    )
    args = parser.parse_args()

    if args.post_id:
        # Test specific question
        await test_forecast_question(args.post_id, args.num_runs)
    else:
        # Test all example questions
        print("No --post-id provided, testing example questions...")
        print(f"Example questions: {EXAMPLE_QUESTIONS}")
        
        for question_id, post_id in EXAMPLE_QUESTIONS:
            try:
                await test_forecast_question(post_id, args.num_runs)
            except Exception as e:
                print(f"Failed on post {post_id}: {e}")
                continue


if __name__ == "__main__":
    asyncio.run(main())
