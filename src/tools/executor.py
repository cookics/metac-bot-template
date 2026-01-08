"""
Tool Execution Loop for Agentic Tool Calling.

Handles the model → tool → model cycle:
1. Model receives prompt with available tools
2. Model responds with tool_calls
3. We execute tools locally and collect results
4. We append tool results as role: tool messages
5. We call model again with updated messages
6. Loop until model responds without tool calls
"""
import json
import asyncio
from typing import Optional
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENROUTER_API_KEY,
    RESEARCH_MODEL,
    RESEARCH_TEMP,
    llm_rate_limiter
)
from .base import BaseTool, ToolResult


@dataclass
class ToolCall:
    """Represents a single tool call from the model."""
    id: str
    name: str
    arguments: dict
    

async def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str = RESEARCH_MODEL,
    temperature: float = RESEARCH_TEMP,
    tool_choice: str = "auto",
    thinking: bool = False
) -> ChatCompletionMessage:
    """
    Call LLM with tool definitions via OpenRouter.
    
    Args:
        messages: Conversation messages
        tools: List of OpenRouter-compatible tool schemas
        model: Model to use
        temperature: Sampling temperature
        tool_choice: "auto", "none", or specific tool
    
    Returns:
        ChatCompletionMessage with potential tool_calls
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
        try:
            print(f"[Tool Executor] Calling {model} with {len(tools)} tools available (Thinking: {thinking})")
            
            # Prepare extra_body for thinking if requested
            extra_body = {}
            if thinking:
                from config import REASONING_MAX_TOKENS
                extra_body["reasoning"] = {
                    "max_tokens": REASONING_MAX_TOKENS
                }

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice=tool_choice if tools else None,
                temperature=temperature,
                extra_body=extra_body if extra_body else None
                # Thinking enabled if requested (essential for Gemini 3 forecast/tool use)
            )
            
            message = response.choices[0].message
            generation_id = response.id if hasattr(response, 'id') else None
            usage = response.usage if hasattr(response, 'usage') else None
            
            if message.tool_calls:
                print(f"[Tool Executor] Model requested {len(message.tool_calls)} tool call(s)")
                for tc in message.tool_calls:
                    print(f"  - {tc.function.name}({tc.function.arguments[:100]}...)")
            else:
                print(f"[Tool Executor] Model finished (no tool calls)")
            
            # Attach usage info to message for cost tracking
            message._generation_id = generation_id
            message._usage = usage
            
            return message
            
        except Exception as e:
            print(f"[Tool Executor] Error calling LLM: {str(e)}")
            raise


def parse_tool_calls(message: ChatCompletionMessage) -> list[ToolCall]:
    """Extract tool calls from a ChatCompletionMessage."""
    if not message.tool_calls:
        return []
    
    calls = []
    for tc in message.tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {"raw": tc.function.arguments}
        
        calls.append(ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=args
        ))
    
    return calls


async def execute_tool(tool: BaseTool, arguments: dict) -> ToolResult:
    """Execute a single tool with given arguments."""
    try:
        print(f"[Tool Executor] Executing {tool.name} with args: {arguments}")
        result = await tool.execute(**arguments)
        print(f"[Tool Executor] {tool.name} completed: success={result.success}")
        return result
    except Exception as e:
        print(f"[Tool Executor] {tool.name} failed with error: {str(e)}")
        return ToolResult(
            success=False,
            data=None,
            error=str(e)
        )


async def run_tool_calling_loop(
    initial_prompt: str,
    tools: list[BaseTool],
    model: str = RESEARCH_MODEL,
    temperature: float = RESEARCH_TEMP,
    max_iterations: int = 5,
    system_prompt: Optional[str] = None,
    thinking: bool = False
) -> tuple[str, list[dict], list[dict]]:
    """
    Run the complete tool calling loop until the model stops calling tools.
    
    Args:
        initial_prompt: The user's question/prompt
        tools: List of BaseTool instances available to the model
        model: LLM model to use
        temperature: Sampling temperature
        max_iterations: Max number of tool-calling rounds
        system_prompt: Optional system prompt
    
    Returns:
        Tuple of (final_response_text, list_of_all_tool_results, list_of_all_messages)
    """
    # Build tool registry and schemas
    tool_registry = {t.name: t for t in tools}
    tool_schemas = [t.to_openrouter_schema() for t in tools]
    
    # Initialize messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": initial_prompt})
    
    all_tool_results = []
    
    for iteration in range(max_iterations):
        print(f"\n[Tool Executor] === Iteration {iteration + 1}/{max_iterations} ===")
        
        # Call model with tools
        response = await call_llm_with_tools(
            messages=messages,
            tools=tool_schemas,
            model=model,
            temperature=temperature,
            thinking=thinking
        )
        
        # Parse tool calls
        tool_calls = parse_tool_calls(response)
        
        if not tool_calls:
            # Model is done - return its final response
            final_text = response.content or ""
            print(f"[Tool Executor] Loop complete after {iteration + 1} iteration(s)")
            return final_text, all_tool_results, messages
        
        # Add assistant message with tool calls to history
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]
        })
        
        # Execute each tool call
        for tc in tool_calls:
            tool = tool_registry.get(tc.name)
            
            if not tool:
                result = ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown tool: {tc.name}"
                )
            else:
                result = await execute_tool(tool, tc.arguments)
            
            # Store result
            all_tool_results.append({
                "tool_call_id": tc.id,
                "tool_name": tc.name,
                "arguments": tc.arguments,
                "result": result.data if result.success else None,
                "error": result.error
            })
            
            # Add tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result.to_message_content()
            })
    
    # Max iterations reached
    print(f"[Tool Executor] Max iterations ({max_iterations}) reached")
    last_content = messages[-1].get("content", "") if messages else ""
    return last_content, all_tool_results, messages


# Convenience function for running tools in parallel
async def execute_tools_parallel(
    tools: list[BaseTool],
    tool_calls: list[ToolCall]
) -> list[ToolResult]:
    """Execute multiple tool calls in parallel."""
    tool_registry = {t.name: t for t in tools}
    
    async def run_one(tc: ToolCall) -> ToolResult:
        tool = tool_registry.get(tc.name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"Unknown tool: {tc.name}")
        return await execute_tool(tool, tc.arguments)
    
    results = await asyncio.gather(*[run_one(tc) for tc in tool_calls])
    return list(results)
