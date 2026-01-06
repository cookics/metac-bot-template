"""
Prompt templates for forecasting.
These are the templates you'll iterate on to improve forecast quality.
"""

# ========================= RESEARCH AGENT PROMPT =========================

RESEARCH_AGENT_PROMPT = """
You are a research assistant helping a forecaster answer questions.

The forecaster needs to answer this question:
{question}

Below are search results from the web. Your job is to:
1. Review each result and determine if it is RELEVANT to answering the question
2. Select only the most relevant results (usually 3-5, but use your judgment)
3. Write a brief summary of the key findings that would help answer the question

Search Results (JSON):
{results_json}

Respond in EXACTLY this format:

RELEVANT_INDICES: [list the index numbers of relevant results, e.g., 0, 2, 5]
SUMMARY: Write a 2-4 sentence summary of the key information from the relevant results that would help forecast this question. Focus on facts, data, and recent developments.
"""

LINK_ANALYSIS_PROMPT = """
You are a research assistant analyzing search results to find additional useful sources.

The forecaster is trying to answer this question:
{question}

Below are search results that were already retrieved. Your job is to look through the content and identify any LINKS or URLS mentioned within the text that might provide additional valuable data for answering the question.

Think about:
- What kind of data would be most useful for this forecast? (statistics, official reports, recent news, expert analysis)
- Which mentioned links might contain that data?
- Only select links that are likely to have NEW information not already in the search results

Search Results Content:
{results_content}

Select UP TO 4 of the most promising URLs to crawl. Only select links that:
1. Are mentioned in the search result text (not the source URLs themselves)
2. Likely contain data, statistics, or expert analysis relevant to the question
3. Are from reputable sources (government sites, research institutions, major news outlets)

If no additional links are worth crawling, return an empty list.

Respond in EXACTLY this format:
REASONING: Brief explanation of what data you're looking for and why these links might help
URLS_TO_CRAWL: ["url1", "url2", "url3"]
"""

SELF_ASSESSMENT_PROMPT = """
You just made a forecast. Now honestly assess your work:

Question: {question}
Your Forecast: {forecast}

Rate the following on a scale of 1-10 and explain briefly:

1. **Information Quality** (1-10): How much relevant, reliable data did you have?
   - 1-3: Very little info, mostly guessing
   - 4-6: Some relevant data but gaps
   - 7-10: Rich, recent, authoritative sources

2. **Reasoning Depth** (1-10): How thorough was your analysis?
   - 1-3: Surface level, didn't consider alternatives
   - 4-6: Considered main factors
   - 7-10: Deep analysis, multiple scenarios, calibrated

3. **Confidence** (1-10): How sure are you of this forecast?
   - 1-3: Very uncertain, could easily be wrong
   - 4-6: Moderate confidence
   - 7-10: High confidence based on strong evidence

Respond in this format:
INFO_QUALITY: [1-10] - [brief reason]
REASONING_DEPTH: [1-10] - [brief reason]  
CONFIDENCE: [1-10] - [brief reason]
"""

# ========================= FORECASTING PROMPTS =========================

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
