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
USE_SMART_SEARCHER = False  # Set to True to use forecasting-tools SmartSearcher instead of pure Exa search
USE_TOOLS = True  # Set to True to enable agentic tool-calling during research

# ========================= API KEYS =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For EXA Smart Searcher

# ========================= LLM SETTINGS =========================
# Supported providers: "openrouter", "metaculus_proxy"
LLM_PROVIDER = "openrouter"

# ========================= MODE SELECTION =========================
#Toggle between EXPENSIVE (high quality) and CHEAP (cost-effective) mode
EXPENSIVE_MODE = False  # Set to True for premium models, False for budget models

# ========================= MODEL PRESETS =========================
# ALL MODES:
#   - Research: x-ai/grok-4.1-fast      (ALWAYS THINKS on OpenRouter)

# Global reasoning effort for models with thinking modes (Gemini 3, Grok, Claude 4.5)
REASONING_EFFORT = "medium" # Options: low, medium, high
REASONING_MAX_TOKENS = 4096 # Minimum is usually 1024 for Claude models

# CLAUDE 4.5 OPUS THINKING TOGGLE
# By default, Claude 4.5 Opus does NOT use extended thinking (to save cost/tokens).
# Set this to True to explicitly EXPOSE and ENABLE the thinking mode for Opus.
CLAUDE_OPUS_THINKING_ENABLED = False 

if EXPENSIVE_MODE:
    # --- EXPENSIVE MODE MODELS ---
    # Grok 4.1 Fast: Excellent for tool-calling, fast context processing.
    # Thinking: DISABLED for cost efficiency (set RESEARCH_THINKING = False)
    RESEARCH_MODEL = "x-ai/grok-4.1-fast"
    RESEARCH_TEMP = 0.6
    RESEARCH_THINKING = False  # Disabled to reduce token usage
    
    # Claude 4.5 Opus: Frontier reasoning model.
    # Thinking: DISABLED for cost efficiency (CLAUDE_OPUS_THINKING_ENABLED = False)
    FORECAST_MODEL = "anthropic/claude-opus-4.5"
    FORECAST_TEMP = 1.0  # Preferred for the new Claude models
    FORECAST_THINKING = False  # Explicitly disabled
else:
    # --- CHEAP MODE MODELS ---
    # Grok 4.1 Fast: Standard fast research (Replacing Gemini 2.0)
    RESEARCH_MODEL = "x-ai/grok-4.1-fast"
    RESEARCH_TEMP = 0.6
    RESEARCH_THINKING = False
    
    # Gemini 3 Flash Preview: Reasoning-enabled forecaster
    # Thinking: ENABLED (user requested 'gemini flash thiking')
    FORECAST_MODEL = "google/gemini-3-flash-preview"
    FORECAST_TEMP = 1.0 # High temp works well with reasoning
    FORECAST_THINKING = True

# Legacy settings (for backward compatibility)
OPENROUTER_MODEL = FORECAST_MODEL
OPENROUTER_TEMP = FORECAST_TEMP

METACULUS_PROXY_MODEL = "gpt-4o"
METACULUS_PROXY_TEMP = 0.7

# Derived defaults (uses forecast model as default)
if LLM_PROVIDER == "openrouter":
    DEFAULT_MODEL = FORECAST_MODEL
    DEFAULT_TEMP = FORECAST_TEMP
else:
    DEFAULT_MODEL = METACULUS_PROXY_MODEL
    DEFAULT_TEMP = METACULUS_PROXY_TEMP

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
