import asyncio
from openai import AsyncOpenAI
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Get OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Define the concurrent request limit
CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

async def call_llm(prompt: str, model: str = "google/gemini-2.0-flash-thinking-exp:free", temperature: float = 0.9) -> str:
    """
    Makes a streaming completion request to OpenRouter's API with concurrent request limiting.
    Uses the Gemini 2.0 Flash Thinking Experimental 01-21 model.
    """
    # Initialize the OpenRouter client
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
                top_p=1.0,  # Supported parameter per OpenRouter docs
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

# Example usage
async def test_call_llm():
    test_prompt = "Hello! "
    try:
        response = await call_llm(test_prompt)
        print(f"Test completed. Final response: {response}")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_call_llm())