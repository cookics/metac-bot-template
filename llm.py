"""
LLM client wrappers for OpenRouter and OpenAI.
These are stable once configured and rarely need changes.
"""
from openai import AsyncOpenAI
from config import OPENROUTER_API_KEY, METACULUS_TOKEN, llm_rate_limiter


async def call_llm(
    prompt: str, 
    model: str = "google/gemini-2.0-flash-001", 
    temperature: float = 0.9
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

    async with llm_rate_limiter:
        collected_content = []
        try:
            print(f"Sending request to OpenRouter with model: {model}")
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1.0,
                stream=True,
            )

            print("Receiving streamed response...")
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    collected_content.append(chunk.choices[0].delta.content)
                    print(f"Chunk received: {chunk.choices[0].delta.content}")

            result = "".join(collected_content)
            print(f"Full response: {result}")
            return result

        except Exception as e:
            print(f"Error in call_llm: {str(e)}")
            raise


async def call_llm_oai(
    prompt: str, 
    model: str = "gpt-4o", 
    temperature: float = 0.3
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
