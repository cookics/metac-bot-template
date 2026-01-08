"""
LLM client wrappers for OpenRouter and OpenAI.
These are stable once configured and rarely need changes.
"""
from openai import AsyncOpenAI
from config import (
    OPENROUTER_API_KEY, 
    METACULUS_TOKEN, 
    llm_rate_limiter,
    LLM_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_TEMP,
    METACULUS_PROXY_MODEL,
    METACULUS_PROXY_TEMP,
    REASONING_EFFORT,
    REASONING_MAX_TOKENS
)


async def call_llm(
    prompt: str, 
    model: str = DEFAULT_MODEL, 
    temperature: float = DEFAULT_TEMP,
    provider: str = LLM_PROVIDER,
    thinking: bool = False,
    return_stats: bool = False
) -> str | tuple[str, dict]:
    """
    Unified entry point for LLM calls. Routes to OpenRouter or Metaculus Proxy.
    
    If return_stats=True and using OpenRouter, returns (response, stats_dict).
    """
    if provider == "openrouter":
        return await call_llm_openrouter(prompt, model, temperature, thinking, return_stats)
    elif provider == "metaculus_proxy":
        return await call_llm_metaculus_proxy(prompt, model, temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def call_llm_openrouter(
    prompt: str, 
    model: str = "google/gemini-2.0-flash-001", 
    temperature: float = 0.9,
    thinking: bool = False,
    return_stats: bool = False
) -> str | tuple[str, dict]:
    """
    Makes a streaming completion request to OpenRouter's API with rate limiting.
    
    If return_stats=True, returns (response, stats_dict) where stats_dict contains
    generation_id, tokens, and cost info.
    """
    import httpx
    
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        max_retries=2,
    )

    extra_body = {}
    if thinking:
        # OpenRouter only allows ONE of 'effort' or 'max_tokens'
        # We use max_tokens for strict wiring as requested
        extra_body["reasoning"] = {
            "max_tokens": REASONING_MAX_TOKENS
        }

    async with llm_rate_limiter:
        collected_content = []
        generation_id = None
        try:
            print(f"Sending request to OpenRouter with model: {model} (Thinking: {thinking})")
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1.0,
                stream=True,
                extra_body=extra_body
            )

            print("Receiving streamed response...")
            async for chunk in stream:
                # Capture the generation ID from first chunk
                if generation_id is None and hasattr(chunk, 'id'):
                    generation_id = chunk.id
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    collected_content.append(chunk.choices[0].delta.content)

            result = "".join(collected_content)
            
            if return_stats and generation_id:
                stats = await fetch_generation_stats(generation_id)
                return result, stats
            
            return result

        except Exception as e:
            print(f"Error in call_llm_openrouter: {str(e)}")
            raise


async def fetch_generation_stats(generation_id: str) -> dict:
    """
    Fetch generation statistics (native tokens, cost) from OpenRouter.
    
    Uses the /api/v1/generation endpoint to get accurate token counts
    and costs based on the model's native tokenizer.
    """
    import httpx
    import asyncio
    
    url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    
    # Wait a moment for OpenRouter to process the generation
    await asyncio.sleep(0.5)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                data = response.json().get("data", {})
                return {
                    "generation_id": generation_id,
                    "native_tokens_prompt": data.get("native_tokens_prompt", 0),
                    "native_tokens_completion": data.get("native_tokens_completion", 0),
                    "total_cost": data.get("total_cost", 0.0),
                    "model": data.get("model", ""),
                    "usage": data.get("usage", {}),
                }
            else:
                print(f"[Stats] Failed to fetch generation stats: {response.status_code}")
                return {"generation_id": generation_id, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        print(f"[Stats] Error fetching generation stats: {e}")
        return {"generation_id": generation_id, "error": str(e)}


async def call_llm_metaculus_proxy(
    prompt: str, 
    model: str = METACULUS_PROXY_MODEL, 
    temperature: float = METACULUS_PROXY_TEMP
) -> str:
    """
    Makes a streaming completion request to OpenAI via Metaculus proxy.
    """
    client = AsyncOpenAI(
        base_url="https://llm-proxy.metaculus.com/proxy/openai/v1",
        default_headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {METACULUS_TOKEN}",
        },
        api_key="placeholder",  # Required by openai package but not used
        max_retries=2,
    )

    async with llm_rate_limiter:
        collected_content = []
        try:
            print(f"Sending request to Metaculus Proxy with model: {model}")
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    collected_content.append(chunk.choices[0].delta.content)

            return "".join(collected_content)
        except Exception as e:
            print(f"Error in call_llm_metaculus_proxy: {str(e)}")
            raise


# Backward compatibility if needed
async def call_llm_oai(prompt: str, model: str = METACULUS_PROXY_MODEL, temperature: float = METACULUS_PROXY_TEMP) -> str:
    return await call_llm_metaculus_proxy(prompt, model, temperature)
