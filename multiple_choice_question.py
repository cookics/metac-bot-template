import asyncio
import datetime
import re
import os
import numpy as np # Though not explicitly used in the provided functions, it's good practice if similar logic evolves
from llm_service import call_llm
from news import get_research_report
from config import GET_NEWS

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}


Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""

def extract_option_probabilities_from_response(forecast_text: str, options: list[str]) -> list[float]:
    def extract_option_probabilities_raw(text: str) -> list[float]:
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?" # handles integers and floats, with commas
        results = []
        for line in text.split("\n"):
            numbers_in_line = re.findall(number_pattern, line)
            # Only consider lines that seem to be assigning a probability to an option
            # This is a heuristic: checks if line starts with "Option_" or similar, or just looks for numbers
            if numbers_in_line: # Simplified: just take the last number if any found
                try:
                    # Take the last number, assuming it's the probability
                    num_str = numbers_in_line[-1].replace(",", "")
                    results.append(float(num_str) if "." in num_str else int(num_str))
                except ValueError:
                    # Handle cases where conversion might fail, though regex should ensure valid numbers
                    pass 
        return results

    option_probabilities = extract_option_probabilities_raw(forecast_text)
    
    NUM_OPTIONS = len(options)
    # Heuristic: if we got more numbers than options, take the last NUM_OPTIONS
    if len(option_probabilities) >= NUM_OPTIONS:
        return option_probabilities[-NUM_OPTIONS:]
    elif option_probabilities: # Not enough numbers, this is likely an error or unexpected format
        # Pad with zeros or raise error - current implementation implies it expects enough numbers
        # For now, let's return what we have, generate_multiple_choice_forecast will handle mismatch
        return option_probabilities 
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")

def generate_multiple_choice_forecast(options: list[str], option_probabilities: list[float]) -> dict[str, float]:
    if len(options) != len(option_probabilities):
        # If probabilities are fewer than options, pad with a small default value before normalization
        # This is a design choice: alternatively, raise an error.
        # Padding helps avoid errors if LLM provides incomplete list, but might skew results.
        if len(option_probabilities) < len(options) and len(option_probabilities) > 0 :
             print(f"Warning: Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)}). Padding with small values.")
             diff = len(options) - len(option_probabilities)
             option_probabilities.extend([0.01] * diff) # Pad with a small probability
        else: # len_probs > len_options or len_probs == 0
            raise ValueError(
                f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
            )

    # Normalize probabilities to sum to 1, ensuring each is between 0.01 and 0.99
    total_sum = sum(option_probabilities)
    if total_sum == 0: # Avoid division by zero if all probabilities are zero
        # Assign equal probability if sum is zero, or handle as error
        print("Warning: Sum of probabilities is zero. Assigning equal probability.")
        normalized_probabilities = [1.0 / len(options) for _ in option_probabilities]
    else:
        # Ensure positive probabilities before normalization
        positive_probabilities = [max(p, 0) for p in option_probabilities]
        total_sum = sum(positive_probabilities)
        if total_sum == 0: # if all were zero or negative
             normalized_probabilities = [1.0 / len(options) for _ in positive_probabilities]
        else:
            normalized_probabilities = [p / total_sum for p in positive_probabilities]

    # Clamp and re-normalize
    clamped_list = [max(min(x, 0.99), 0.01) for x in normalized_probabilities]
    clamped_sum = sum(clamped_list)
    final_probabilities = [x / clamped_sum for x in clamped_list]
    
    # Adjust for any small floating-point errors to ensure sum is exactly 1
    adjustment = 1.0 - sum(final_probabilities)
    if final_probabilities:
        final_probabilities[-1] += adjustment

    probability_yes_per_category = {}
    for i, option_label in enumerate(options):
        probability_yes_per_category[option_label] = final_probabilities[i]
        
    return probability_yes_per_category

async def get_multiple_choice_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[dict[str, float], str]:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    options = question_details["options"] # list of strings

    default_provider = "asknews"
    if os.getenv("EXA_API_KEY"):
        default_provider = "exa"
    elif os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
        default_provider = "asknews"
    elif os.getenv("PERPLEXITY_API_KEY"):
        default_provider = "perplexity"
    provider = os.getenv("GET_NEWS_PROVIDER", default_provider)
    summary_report = get_research_report(title, provider=provider, enable_research=GET_NEWS)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title, today=today, background=background, resolution_criteria=resolution_criteria,
        fine_print=fine_print, summary_report=summary_report, options=options,
    )

    async def ask_llm_for_multiple_choice_probabilities(content_for_llm: str) -> tuple[dict[str, float], str]:
        rationale = await call_llm(content_for_llm)
        extracted_probabilities = extract_option_probabilities_from_response(rationale, options)
        comment_text = f"EXTRACTED_PROBABILITIES: {extracted_probabilities}\n\nGPT's Answer: {rationale}\n\n\n"
        
        probability_yes_per_category = generate_multiple_choice_forecast(options, extracted_probabilities)
        return probability_yes_per_category, comment_text

    probability_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [f"## Rationale {i+1}\n{comment_text}" for i, comment_text in enumerate(comments)]
    
    # Averaging probabilities across runs
    # Initialize a dictionary to sum probabilities for each option
    summed_probabilities_per_category: dict[str, float] = {option: 0.0 for option in options}
    
    for prob_dict, _ in probability_and_comment_pairs:
        for option, prob in prob_dict.items():
            summed_probabilities_per_category[option] += prob
            
    # Calculate average probabilities
    average_probability_yes_per_category: dict[str, float] = {
        option: total_prob / num_runs 
        for option, total_prob in summed_probabilities_per_category.items()
    }
    
    # Re-normalize the averaged probabilities to ensure they sum to 1
    # This is important because simple averaging might not preserve the sum-to-one property perfectly
    # if individual runs were already normalized.
    total_avg_prob = sum(average_probability_yes_per_category.values())
    if total_avg_prob > 0 : # Avoid division by zero
        final_averaged_probabilities = {
            option: prob / total_avg_prob
            for option, prob in average_probability_yes_per_category.items()
        }
    else: # If sum is zero (e.g. all options got zero prob), distribute equally.
         final_averaged_probabilities = {option: 1.0/len(options) for option in options}


    final_comment = f"Average Probability Yes Per Category: `{final_averaged_probabilities}`\n\n" + "\n\n".join(final_comment_sections)
    return final_averaged_probabilities, final_comment
