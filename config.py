import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# General Settings
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = 5  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
GET_NEWS = True  # set to True to enable the bot to do online research

# API Keys - These are for services used directly by main.py or broadly
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
# OPENAI_API_KEY is also used by news.py for Exa's SmartSearcher, so it's defined here.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Note: ASKNEWS_CLIENT_ID, ASKNEWS_SECRET, EXA_API_KEY, PERPLEXITY_API_KEY are loaded in news.py

# Tournament IDs
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
AXC_2025_TOURNAMENT_ID = 32564
GIVEWELL_ID = 3600
RESPIRATORY_OUTLOOK_ID = 3411
Q2_2025_AI_BENCHMARKING_ID = 32721

TOURNAMENT_ID = Q2_2025_AI_BENCHMARKING_ID  # Default tournament

# Example Questions
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (578, 578),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    (14333, 14333),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    (22427, 22427),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
]

# Metaculus API Configuration
API_BASE_URL = "https://www.metaculus.com/api"
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}

# Concurrency Limits
# Note: CONCURRENT_REQUESTS_LIMIT appears twice in main.py, one for llm_rate_limiter and one globally.
# Consolidating to one definition here. If they were meant to be different, this might need adjustment.
CONCURRENT_REQUESTS_LIMIT = 5

# LLM Rate Limiter (initialized in main.py using CONCURRENT_REQUESTS_LIMIT)
# llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT) # This needs asyncio, better to init in main.py
