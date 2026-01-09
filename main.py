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
from pathlib import Path
import json
import re

# Add src to sys.path so modules can find each other
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

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
    ACTIVE_TOURNAMENTS,
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


def save_question_record(tournament_id: str, result: dict, forecast: any, comment: str) -> None:
    """
    Save a record of the forecast in a tournament-specific folder.
    """
    # Create the tournament directory at the root
    tournament_dir = ROOT_DIR / str(tournament_id)
    tournament_dir.mkdir(exist_ok=True)

    # Use a safe filename from the title
    safe_title = re.sub(r'[^\w\s-]', '', result["title"]).strip().replace(" ", "_")[:50]
    filename = f"{safe_title}_{result['url'].split('/')[-2]}.md"
    file_path = tournament_dir / filename

    content = f"# {result['title']}\n\n"
    content += f"**URL:** {result['url']}\n"
    content += f"**Type:** {result['type']}\n\n"
    content += f"## Forecast\n"
    if result["type"] in ["numeric", "date"] and isinstance(forecast, list):
         content += f"CDF: `[{forecast[0]:.4f}, ..., {forecast[-1]:.4f}]` ({len(forecast)} points)\n"
    else:
         content += f"{forecast}\n"
    
    content += f"\n## Rationale\n\n{comment}\n"

    # Add Trace Data
    if result.get("trace"):
        trace = result["trace"]
        content += "\n<details>\n<summary>🔍 Detailed Trace (Research & Forecaster Input)</summary>\n\n"
        
        if trace.get("research_data"):
            content += "### Research Data Provided to Forecaster\n"
            content += f"```text\n{trace['research_data']}\n```\n\n"
            
        if trace.get("forecaster_prompt"):
            content += "### Full Forecaster Prompt\n"
            content += f"```text\n{trace['forecaster_prompt']}\n```\n\n"
            
        if trace.get("research_messages"):
            content += "### Research Agent Conversation Trace\n"
            content += "```json\n" + json.dumps(trace["research_messages"], indent=2, default=str) + "\n```\n\n"
            
        if trace.get("forecaster_messages"):
            content += "### Forecaster Agent Conversation Trace\n"
            content += "```json\n" + json.dumps(trace["forecaster_messages"], indent=2, default=str) + "\n```\n\n"
            
        content += "</details>\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved record to {file_path}")


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
    tournament_id: str = "unknown",
) -> dict:
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

    result = {
        "title": title,
        "url": f"https://www.metaculus.com/questions/{post_id}/",
        "type": question_type,
        "status": "Checked",
        "forecast": "-",
    }

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        result["status"] = "Skipped (Already Made)"
        return result

    if question_type == "binary":
        forecast, comment, trace = await get_binary_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "numeric" or question_type == "date":
        # Numeric and Date questions use the same handler now
        forecast, comment, trace = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
        if trace.get("exa_cost", 0) > 0:
            summary_of_forecast += f"Exa Cost: ${trace['exa_cost']:.4f}\n"
            
    elif question_type == "multiple_choice":
        forecast, comment, trace = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    result["trace"] = trace

    print(f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n")
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    if question_type == "numeric" or question_type == "date":
        summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    if submit_prediction:
        forecast_payload = create_forecast_payload(forecast, question_type)
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"
        result["status"] = "Forecasted & Posted"
    else:
        result["status"] = "Forecasted (Not Posted)"

    # Save record locally regardless of submission
    try:
        save_question_record(tournament_id, result, forecast, comment)
    except Exception as e:
        print(f"Error saving local record for {title}: {e}")

    result["forecast"] = str(forecast)[:100] + "..." if len(str(forecast)) > 100 else str(forecast)
    return result


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
        for question_id, post_id, _ in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id, _ = question_id_post_id
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
        # We still want to write the summary even if some failed
        # raise RuntimeError(error_message)

    generate_github_summary(forecast_summaries, open_question_id_post_id, logs_dir)


def generate_github_summary(results: list, question_info: list, logs_dir: Path) -> None:
    """
    Generate a markdown summary table for GitHub Actions.
    """
    total = len(results)
    forecasted = sum(1 for res in results if not isinstance(res, Exception) and "Forecasted" in res["status"])
    skipped = sum(1 for res in results if not isinstance(res, Exception) and "Skipped" in res["status"])
    errors = sum(1 for res in results if isinstance(res, Exception))

    summary_lines = [
        "## 🤖 Metaculus Bot Run Summary",
        "",
        f"| Statistic | Count |",
        f"| :--- | :--- |",
        f"| **Total Questions Checked** | {total} |",
        f"| **New Forecasts Made** | {forecasted} ✅ |",
        f"| **Sorted/Skipped** | {skipped} ⏭️ |",
        f"| **Errors Encountered** | {errors} {'❌' if errors > 0 else '✅'} |",
        "",
        "### Detailed Results",
        "",
        "| Question | Type | Status | Forecast Preview |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for info, res in zip(question_info, results):
        if isinstance(res, Exception):
            title = info[2]
            url = f"https://www.metaculus.com/questions/{info[1]}/"
            status = f"❌ Error: {res.__class__.__name__}"
            forecast = "-"
        else:
            title = res["title"]
            url = res["url"]
            status = res["status"]
            forecast = f"`{res['forecast']}`"

        summary_lines.append(f"| [{title}]({url}) | {info[1] if isinstance(res, Exception) else res['type']} | {status} | {forecast} |")

    with open(logs_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    with open(logs_dir / "forecast_count.txt", "w", encoding="utf-8") as f:
        f.write(str(forecasted))
        
    print(f"\nWritten summary to {logs_dir / 'summary.md'} and count to {logs_dir / 'forecast_count.txt'}")


async def run_bot(args, logs_dir):
    """
    Main bot orchestration loop.
    """
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
        print(f"Using example questions: {open_question_id_post_id}")
    else:
        # We'll fetch questions per tournament inside the loop if not check-only
        pass

    if args.check_only:
        # Just check if there are questions that need forecasting across all tournaments
        total_needs_forecast = 0
        
        for tournament_id in ACTIVE_TOURNAMENTS:
            print(f"Checking tournament: {tournament_id}")
            tournament_questions = get_open_question_ids_from_tournament(tournament_id=tournament_id)
            
            if SKIP_PREVIOUSLY_FORECASTED_QUESTIONS:
                needs_forecast_count = 0
                for question_id, post_id, title in tournament_questions:
                    post_details = get_post_details(post_id)
                    if not forecast_is_already_made(post_details):
                        needs_forecast_count += 1
                total_needs_forecast += needs_forecast_count
                print(f"  - {needs_forecast_count} questions need forecasting in {tournament_id}")
            else:
                total_needs_forecast += len(tournament_questions)
                print(f"  - {len(tournament_questions)} questions to forecast in {tournament_id}")
        
        # Write output for GitHub Actions
        with open("needs_forecast.txt", "w") as f:
            f.write("true" if total_needs_forecast > 0 else "false")
        
        if total_needs_forecast > 0:
            print(f"✅ Total {total_needs_forecast} questions need forecasting")
        else:
            print("⏭️ No questions need forecasting across any tournament")
    else:
        # Normal forecasting mode
        all_forecast_summaries = []
        all_question_info = []

        # If USE_EXAMPLE_QUESTIONS is True, we only run those
        if USE_EXAMPLE_QUESTIONS:
             print(f"\n{'='*20} Processing Example Questions {'='*20}")
             forecast_tasks = [
                forecast_individual_question(
                    qid,
                    pid,
                    SUBMIT_PREDICTION,
                    NUM_RUNS_PER_QUESTION,
                    SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
                    tournament_id="examples",
                )
                for qid, pid in EXAMPLE_QUESTIONS
            ]
             results = await asyncio.gather(*forecast_tasks, return_exceptions=True)
             all_forecast_summaries.extend(results)
             all_question_info.extend([(qid, pid, "Example") for qid, pid in EXAMPLE_QUESTIONS])
        else:
            for tournament_id in ACTIVE_TOURNAMENTS:
                print(f"\n{'='*20} Processing Tournament: {tournament_id} {'='*20}")
                
                # Ensure the tournament directory exists early, even if no forecasts are made
                tournament_dir = ROOT_DIR / str(tournament_id)
                tournament_dir.mkdir(exist_ok=True)
                
                # Fetch questions for this specific tournament
                tournament_questions = get_open_question_ids_from_tournament(tournament_id=tournament_id)
                print(f"Found {len(tournament_questions)} open questions in {tournament_id}")
                
                if not tournament_questions:
                    continue

                # Run forecasting for this tournament
                forecast_tasks = [
                    forecast_individual_question(
                        qid,
                        pid,
                        SUBMIT_PREDICTION,
                        NUM_RUNS_PER_QUESTION,
                        SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
                        tournament_id=tournament_id,
                    )
                    for qid, pid, title in tournament_questions
                ]
                
                results = await asyncio.gather(*forecast_tasks, return_exceptions=True)
                
                # Print results for this tournament
                for qinfo, res in zip(tournament_questions, results):
                    if isinstance(res, Exception):
                        print(f"Error in {qinfo[0]}: {res}")
                    else:
                        print(f"Status for {res['title']}: {res['status']}")

                all_forecast_summaries.extend(results)
                all_question_info.extend(tournament_questions)

        print("\n", "#" * 100, "\nForecast Summaries (All Tournaments)\n", "#" * 100)
        # Final output reporting
        generate_github_summary(all_forecast_summaries, all_question_info, logs_dir)


if __name__ == "__main__":
    import argparse
    from config import SUBMIT_PREDICTION, NUM_RUNS_PER_QUESTION
    
    parser = argparse.ArgumentParser(description="Metaculus Forecasting Bot")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if forecasting is needed, don't actually forecast"
    )
    args = parser.parse_args()
    
    # Setup log paths
    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("Starting BOT")
    asyncio.run(run_bot(args, logs_dir))

