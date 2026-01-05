"""
Configuration and constants for the Metaculus forecasting bot.
This file contains settings that rarely change.
"""
import os
import asyncio
import dotenv

dotenv.load_dotenv()

# ========================= RUNTIME FLAGS =========================
SUBMIT_PREDICTION = True   # Set to True to publish predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # Set to True to forecast example questions
NUM_RUNS_PER_QUESTION = 1  # The median forecast is taken between runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
GET_NEWS = True  # Set to True to enable EXA research

# ========================= API KEYS =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For EXA Smart Searcher

# ========================= METACULUS API =========================
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"

# ========================= RATE LIMITING =========================
CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

# ========================= TOURNAMENT IDS =========================
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
AXC_2025_TOURNAMENT_ID = 32564
GIVEWELL_ID = 3600
RESPIRATORY_OUTLOOK_ID = 3411
Q2_2025_AI_BENCHMARKING_ID = 32721
SPRING_BOT_BENCH = "spring-aib-2026"

# Active tournament
TOURNAMENT_ID = SPRING_BOT_BENCH

# ========================= EXAMPLE QUESTIONS =========================
# For testing - (question_id, post_id)
EXAMPLE_QUESTIONS = [
    (578, 578),      # Human Extinction - Binary
    (14333, 14333),  # Age of Oldest Human - Numeric
    (22427, 22427),  # Number of New Leading AI Labs - Multiple Choice
]
