import asyncio
import datetime
import re
import os # Import os for getenv
import numpy as np
from llm_service import call_llm
from news import get_research_report
from config import GET_NEWS # For the enable_research flag

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    # question_type = question_details["type"] # Not directly used in this function's logic after refactor

    # Determine the provider for research
    default_provider = "asknews"
    if os.getenv("EXA_API_KEY"): # Checks if EXA_API_KEY is set in environment (loaded by news.py or config.py)
        default_provider = "exa"
    elif os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
        default_provider = "asknews"
    elif os.getenv("PERPLEXITY_API_KEY"):
        default_provider = "perplexity"
    
    provider = os.getenv("GET_NEWS_PROVIDER", default_provider)

    summary_report = get_research_report(title, provider=provider, enable_research=GET_NEWS)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(content_for_llm: str) -> tuple[float, str]:
        rationale = await call_llm(content_for_llm)
        probability = extract_probability_from_response_as_percentage_not_decimal(rationale)
        comment = (
            f"Extracted Probability: {probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )
        return probability, comment

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment_text}" for i, comment_text in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100.0 # Ensure float division

    final_comment = f"Median Probability: {median_probability:.2%}\n\n" + "\n\n".join( # Format as percentage
        final_comment_sections
    )
    return median_probability, final_comment
