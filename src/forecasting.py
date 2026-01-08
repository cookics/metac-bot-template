"""
Forecasting logic - extraction functions and prediction generators.
This is where most forecasting improvements will be made.
"""
import asyncio
import datetime
import re
import numpy as np

from llm import call_llm
from research_agent import run_research_agent, format_results_for_forecaster, run_research_pipeline
from config import FORECAST_MODEL, FORECAST_TEMP, FORECAST_THINKING, USE_TOOLS
from prompts import (
    BINARY_PROMPT_TEMPLATE,
    NUMERIC_PROMPT_TEMPLATE,
    MULTIPLE_CHOICE_PROMPT_TEMPLATE,
)


# ========================= HELPERS =========================

def detect_question_type(title: str) -> str:
    """Detect if a question is a market-related question."""
    market_keywords = [
        "treasury", "yield", "bond", "oas", "spread", "vix", 
        "nvda", "aapl", "msft", "goog", "tsla", "meta", "amzn",
        "stock", "equity", "commodity", "oil", "gold", "dgs10"
    ]
    title_lower = title.lower()
    if any(kw in title_lower for kw in market_keywords):
        return "market"
    return "general"


# ========================= EXTRACTION FUNCTIONS =========================

def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    """Extract probability percentage from LLM response for binary questions."""
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def extract_percentiles_from_response(forecast_text: str) -> dict:
    """Extract percentile values from LLM response for numeric questions."""
    
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
                numbers = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        percentile_values = {}
        for first_num, second_num in results:
            percentile_values[first_num] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def extract_date_percentiles_from_response(forecast_text: str) -> dict:
    """
    Extract percentile values from LLM response for DATE questions.
    Parses dates like "2028-07-01" and converts them to Unix timestamps.
    """
    import calendar
    
    # Pattern to match lines like "Percentile 50: 2028-07-01"
    date_pattern = r"(?:P|p)ercentile\s*(\d+)\s*:\s*(\d{4})-(\d{2})-(\d{2})"
    # Also match simpler format "Percentile 50: 2028" (just year)
    year_only_pattern = r"(?:P|p)ercentile\s*(\d+)\s*:\s*(\d{4})(?!\d|-)"
    
    percentile_values = {}
    
    for line in forecast_text.split("\n"):
        # Try full date match first
        date_match = re.search(date_pattern, line)
        if date_match:
            percentile = int(date_match.group(1))
            year = int(date_match.group(2))
            month = int(date_match.group(3))
            day = int(date_match.group(4))
            try:
                dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
                timestamp = dt.timestamp()
                percentile_values[percentile] = timestamp
            except (ValueError, OverflowError):
                continue
            continue
        
        # Try year-only match
        year_match = re.search(year_only_pattern, line)
        if year_match:
            percentile = int(year_match.group(1))
            year = int(year_match.group(2))
            try:
                # Use July 1st as middle of year
                dt = datetime.datetime(year, 7, 1, tzinfo=datetime.timezone.utc)
                timestamp = dt.timestamp()
                percentile_values[percentile] = timestamp
            except (ValueError, OverflowError):
                continue
    
    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract date predictions from response: {forecast_text}")


def extract_option_probabilities_from_response(forecast_text: str, options) -> list:
    """Extract option probabilities from LLM response for multiple choice questions."""
    
    def extract_option_probabilities(text):
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
        results = []

        for line in text.split("\n"):
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)
    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


# ========================= CDF GENERATION =========================

def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
) -> list[float]:
    """
    Generate a 201-point CDF from percentile values.
    """
    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = lower_bound
    range_max = upper_bound
    range_size = range_max - range_min
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Adjust values at bounds
    for percentile, value in list(percentile_values.items()):
        if not open_lower_bound and value <= range_min + buffer:
            percentile_values[percentile] = range_min + buffer
        if not open_upper_bound and value >= range_max - buffer:
            percentile_values[percentile] = range_max - buffer

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
            percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max
    else:
        percentile_values[100] = range_max

    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
            percentile_values[int(0.5 * percentile_min)] = range_min
    else:
        percentile_values[0] = range_min

    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value

    value_percentiles = {
        value: key for key, value in normalized_percentile_values.items()
    }

    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return [scale(x) for x in np.linspace(0, 1, 201)]

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    # Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for smooth monotonic curves
    # This replaces the jagged linear interpolation with smooth curves
    try:
        from scipy.interpolate import PchipInterpolator
        
        sorted_pairs = sorted(value_percentiles.items())
        known_x = np.array([pair[0] for pair in sorted_pairs])
        known_y = np.array([pair[1] for pair in sorted_pairs])
        
        # PCHIP requires at least 2 points
        if len(known_x) >= 2:
            pchip = PchipInterpolator(known_x, known_y, extrapolate=True)
            continuous_cdf = pchip(cdf_xaxis).tolist()
        else:
            # Fallback to linear for edge cases
            continuous_cdf = [known_y[0] if len(known_y) > 0 else 0.5] * len(cdf_xaxis)
    except ImportError:
        # Fallback to linear interpolation if scipy not available
        def linear_interpolation(x_values, xy_pairs):
            sorted_pairs = sorted(xy_pairs.items())
            known_x = [pair[0] for pair in sorted_pairs]
            known_y = [pair[1] for pair in sorted_pairs]
            y_values = []

            for x in x_values:
                if x in known_x:
                    y_values.append(known_y[known_x.index(x)])
                else:
                    i = 0
                    while i < len(known_x) and known_x[i] < x:
                        i += 1

                    if i == 0:
                        y_values.append(known_y[0])
                    elif i == len(known_x):
                        y_values.append(known_y[-1])
                    else:
                        x0, x1 = known_x[i - 1], known_x[i]
                        y0, y1 = known_y[i - 1], known_y[i]
                        y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                        y_values.append(y)

            return y_values
        continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)
    
    # Monotonicity fix: Ensure strictly increasing by at least 5e-5 (Metaculus requires 5e-05)
    min_increment = 5e-5
    for i in range(1, len(continuous_cdf)):
        if continuous_cdf[i] <= continuous_cdf[i-1] + min_increment:
            continuous_cdf[i] = continuous_cdf[i-1] + min_increment
    
    # Clamp CDF to valid range based on bounds
    # Open bounds require CDF to stay within [0.001, 0.999] to leave room for tails
    min_cdf = 0.001 if open_lower_bound else 0.0
    max_cdf = 0.999 if open_upper_bound else 1.0
    
    # Scale the CDF to fit within bounds while preserving monotonicity
    current_min = min(continuous_cdf)
    current_max = max(continuous_cdf)
    
    if current_max > max_cdf or current_min < min_cdf:
        # Need to scale - use linear scaling to preserve monotonicity
        scale_factor = (max_cdf - min_cdf) / (current_max - current_min + 1e-10)
        for i in range(len(continuous_cdf)):
            continuous_cdf[i] = min_cdf + (continuous_cdf[i] - current_min) * scale_factor
    
    # Final monotonicity check - ensure strictly increasing by at least 5e-5
    for i in range(1, len(continuous_cdf)):
        if continuous_cdf[i] <= continuous_cdf[i-1] + min_increment:
            continuous_cdf[i] = continuous_cdf[i-1] + min_increment
    
    # Final clamp to ensure we don't exceed max after monotonicity adjustment
    for i in range(len(continuous_cdf)):
        continuous_cdf[i] = max(min_cdf, min(max_cdf, continuous_cdf[i]))
            
    return continuous_cdf


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Generate normalized probability distribution for multiple choice.
    """
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]
        total_sum = sum(clamped_list)
        normalized_list = [x / total_sum for x in clamped_list]
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment
        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


# ========================= PREDICTION FUNCTIONS =========================

async def get_binary_gpt_prediction(
    question_details: dict, 
    num_runs: int, 
    research_data: list[dict] = None,
    model: str = None,
    prompt_template: str = None,
    thinking: bool = FORECAST_THINKING
) -> tuple[float, str]:
    """Generate prediction for binary question."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    # Two-agent pipeline: Research Agent filters/summarizes, then Forecasting Agent predicts
    if USE_TOOLS:
        q_type = detect_question_type(title)
        research = await run_research_pipeline(title, question_type=q_type)
        summary_report = research["formatted_for_forecaster"]
    else:
        relevant_results, research_summary = await run_research_agent(title, existing_results=research_data)
        summary_report = format_results_for_forecaster(relevant_results, research_summary)

    template = prompt_template or BINARY_PROMPT_TEMPLATE
    content = template.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(content: str) -> tuple[float, str]:
        rationale = await call_llm(content, model=model or FORECAST_MODEL, temperature=FORECAST_TEMP, thinking=thinking)
        probability = extract_probability_from_response_as_percentage_not_decimal(rationale)
        comment = f"Extracted Probability: {probability}%\n\nGPT's Answer: {rationale}\n\n\n"
        return probability, comment

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    final_comment = f"Median Probability: {median_probability}\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_probability, final_comment


async def get_numeric_gpt_prediction(
    question_details: dict, 
    num_runs: int, 
    research_data: list[dict] = None,
    model: str = None,
    prompt_template: str = None,
    thinking: bool = FORECAST_THINKING
) -> tuple[list[float], str]:
    """Generate prediction for numeric question."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling["zero_point"]

    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    if USE_TOOLS:
        q_type = detect_question_type(title)
        research = await run_research_pipeline(title, question_type=q_type)
        summary_report = research["formatted_for_forecaster"]
        # Capture metadata
        metadata = {
            "exa_cost": research.get("exa_cost", 0.0),
            "tool_usage": research.get("tool_usage", {})
        }
    else:
        relevant_results, research_summary = await run_research_agent(title, existing_results=research_data)
        summary_report = format_results_for_forecaster(relevant_results, research_summary)
        metadata = {"exa_cost": 0.0, "tool_usage": {}}

    template = prompt_template or NUMERIC_PROMPT_TEMPLATE
    content = template.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
    )

    async def ask_llm_to_get_cdf(content: str) -> tuple[list[float], str]:
        # --- NEW LOGIC: USE TOOL LOOP IF AVAILABLE ---
        if USE_TOOLS:
            from tools import get_tool, run_tool_calling_loop
            from prompts import FORECAST_SYSTEM_PROMPT
            
            # Initialize the parametric tool
            tools = [get_tool("get_parametric_cdf")]
            
            # Run the tool loop
            # thinking=True is CRITICAL for Gemini 3 to reason about mean/std before calling the tool
            final_response, tool_calls, messages = await run_tool_calling_loop(
                initial_prompt=content,
                tools=tools,
                model=model or FORECAST_MODEL,
                temperature=FORECAST_TEMP,
                max_iterations=3,
                system_prompt=FORECAST_SYSTEM_PROMPT,
                thinking=thinking
            )
            rationale = final_response
            
            # Capture tool inputs for specific logging
            tool_summary = ""
            for tc in tool_calls:
                if tc["tool_name"] == "get_parametric_cdf":
                    args = tc["arguments"]
                    tool_summary += f"\n[Parametric Tool Used: Mean={args.get('mean')}, Std={args.get('std')}, Skew={args.get('skew', 0)}]\n"
            
            if tool_summary:
                rationale += tool_summary
                
        else:
            # Fallback for no tools
            rationale = await call_llm(content, model=model or FORECAST_MODEL, temperature=FORECAST_TEMP, thinking=thinking)

        # Use date-specific extractor for date questions (converts to timestamps)
        if question_type == "date":
            percentile_values = extract_date_percentiles_from_response(rationale)
        else:
            percentile_values = extract_percentiles_from_response(rationale)

        comment = (
            f"Extracted Percentile_values: {percentile_values}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        cdf = generate_continuous_cdf(
            percentile_values,
            question_type,
            open_upper_bound,
            open_lower_bound,
            upper_bound,
            lower_bound,
            zero_point,
        )
        return cdf, comment

    cdf_and_comment_pairs = await asyncio.gather(
        *[ask_llm_to_get_cdf(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    median_cdf: list[float] = np.median(all_cdfs, axis=0).tolist()

    final_comment = f"Median CDF: `{str(median_cdf)[:100]}...`\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_cdf, final_comment, metadata


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
    research_data: list[dict] = None,
    model: str = None,
    prompt_template: str = None,
    thinking: bool = FORECAST_THINKING
) -> tuple[dict[str, float], str]:
    """Generate prediction for multiple choice question."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    options = question_details["options"]

    if USE_TOOLS:
        q_type = detect_question_type(title)
        research = await run_research_pipeline(title, question_type=q_type)
        summary_report = research["formatted_for_forecaster"]
    else:
        relevant_results, research_summary = await run_research_agent(title, existing_results=research_data)
        summary_report = format_results_for_forecaster(relevant_results, research_summary)

    template = prompt_template or MULTIPLE_CHOICE_PROMPT_TEMPLATE
    content = template.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> tuple[dict[str, float], str]:
        rationale = await call_llm(content, model=model or FORECAST_MODEL, temperature=FORECAST_TEMP, thinking=thinking)

        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            d[option] for d in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return average_probability_yes_per_category, final_comment
