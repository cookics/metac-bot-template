import asyncio
import datetime
import re
import os
import numpy as np
from llm_service import call_llm
from news import get_research_report
from config import GET_NEWS

NUMERIC_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

{lower_bound_message}
{upper_bound_message}


Formatting Instructions:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unkowns.

The last thing you write is your final answer as:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""

def extract_percentiles_from_response(forecast_text: str) -> dict:
    def extract_percentile_numbers(text) -> dict:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        results = []
        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers_float = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers_float) > 1:
                    first_number = numbers_float[0]
                    last_number = numbers_float[-1]
                    if "-" in line.split(":")[-1]: # Check for negative sign for the value
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))
        percentile_values = {}
        for first_num, second_num in results:
            percentile_values[int(first_num)] = second_num # Percentile should be int
        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)
    if not percentile_values: # Check if empty
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")
    return percentile_values

def generate_continuous_cdf(
    percentile_values: dict,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
) -> list[float]:
    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min_val = lower_bound
    range_max_val = upper_bound
    range_size = range_max_val - range_min_val
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Adjust values at bounds
    for percentile, value in list(percentile_values.items()):
        if not open_lower_bound and value <= range_min_val + buffer:
            percentile_values[percentile] = range_min_val + buffer
        if not open_upper_bound and value >= range_max_val - buffer:
            percentile_values[percentile] = range_max_val - buffer
    
    # Set CDF values outside given percentiles
    if open_upper_bound:
        if range_max_val > percentile_values[percentile_max]: # ensure there is a gap to fill
             percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max_val
    else:
        percentile_values[100] = range_max_val

    if open_lower_bound:
        if range_min_val < percentile_values[percentile_min]: # ensure there is a gap to fill
            percentile_values[int(0.5 * percentile_min)] = range_min_val
    else:
        percentile_values[0] = range_min_val
    
    sorted_percentile_values = dict(sorted(percentile_values.items()))

    normalized_percentile_values = {float(k)/100.0: v for k, v in sorted_percentile_values.items()}
    
    value_percentiles = {v: k for k, v in normalized_percentile_values.items()}

    def generate_cdf_locations(min_val, max_val, zp):
        if zp is None:
            scale = lambda x: min_val + (max_val - min_val) * x
        else:
            # Ensure zero_point is not equal to range_min to avoid division by zero
            if zp == min_val: 
                 scale = lambda x: min_val + (max_val - min_val) * x # fallback to linear
            else:
                deriv_ratio = (max_val - zp) / (min_val - zp)
                if deriv_ratio == 1: # fallback to linear if ratio is 1 (e.g. max_val == min_val)
                    scale = lambda x: min_val + (max_val - min_val) * x
                else:
                    scale = lambda x: min_val + (max_val - min_val) * (deriv_ratio**x - 1) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, 201)]

    cdf_xaxis = generate_cdf_locations(range_min_val, range_max_val, zero_point)
    
    # Linear interpolation
    known_x = sorted(list(value_percentiles.keys()))
    known_y = [value_percentiles[x_val] for x_val in known_x]
    
    continuous_cdf = np.interp(cdf_xaxis, known_x, known_y).tolist()
    return continuous_cdf

async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[list[float], str]:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    # question_type = question_details["type"] # Not used
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling.get("zero_point") # Use .get for safety as it might be missing

    upper_bound_message = "" if open_upper_bound else f"The outcome can not be higher than {upper_bound}."
    lower_bound_message = "" if open_lower_bound else f"The outcome can not be lower than {lower_bound}."

    default_provider = "asknews"
    if os.getenv("EXA_API_KEY"):
        default_provider = "exa"
    elif os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
        default_provider = "asknews"
    elif os.getenv("PERPLEXITY_API_KEY"):
        default_provider = "perplexity"
    provider = os.getenv("GET_NEWS_PROVIDER", default_provider)
    summary_report = get_research_report(title, provider=provider, enable_research=GET_NEWS)

    content = NUMERIC_PROMPT_TEMPLATE.format(
        title=title, today=today, background=background, resolution_criteria=resolution_criteria,
        fine_print=fine_print, summary_report=summary_report,
        lower_bound_message=lower_bound_message, upper_bound_message=upper_bound_message,
    )

    async def ask_llm_to_get_cdf(content_for_llm: str) -> tuple[list[float], str]:
        rationale = await call_llm(content_for_llm)
        percentile_values = extract_percentiles_from_response(rationale)
        comment_text = f"Extracted Percentile_values: {percentile_values}\n\nGPT's Answer: {rationale}\n\n\n"
        cdf = generate_continuous_cdf(
            percentile_values, open_upper_bound, open_lower_bound,
            upper_bound, lower_bound, zero_point,
        )
        return cdf, comment_text

    cdf_and_comment_pairs = await asyncio.gather(*[ask_llm_to_get_cdf(content) for _ in range(num_runs)])
    
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [f"## Rationale {i+1}\n{comment_text}" for i, comment_text in enumerate(comments)]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    
    all_cdfs_np = np.array(cdfs)
    median_cdf: list[float] = np.median(all_cdfs_np, axis=0).tolist()
    
    final_comment = f"Median CDF: `{str(median_cdf)[:100]}...`\n\n" + "\n\n".join(final_comment_sections)
    return median_cdf, final_comment
