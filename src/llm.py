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
    thinking: bool = False
) -> str:
    """
    Unified entry point for LLM calls. Routes to OpenRouter or Metaculus Proxy.
    """
    if provider == "openrouter":
        return await call_llm_openrouter(prompt, model, temperature, thinking)
    elif provider == "metaculus_proxy":
        return await call_llm_metaculus_proxy(prompt, model, temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def call_llm_openrouter(
    prompt: str, 
    model: str = "google/gemini-2.0-flash-001", 
    temperature: float = 0.9,
    thinking: bool = False
) -> str:
    """
    Makes a streaming completion request to OpenRouter's API with rate limiting.
    """
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
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    collected_content.append(chunk.choices[0].delta.content)

            result = "".join(collected_content)
            return result

        except Exception as e:
            print(f"Error in call_llm_openrouter: {str(e)}")
            raise


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
