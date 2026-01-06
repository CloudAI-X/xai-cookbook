#!/usr/bin/env python3
"""
05_async_requests.py - Async/Parallel Requests with asyncio

This example demonstrates how to make asynchronous and parallel requests
to the xAI API using asyncio. This is useful for improving throughput
when you need to make multiple independent API calls.

Key concepts:
- Using AsyncOpenAI client
- Making concurrent requests with asyncio.gather
- Async streaming responses
- Performance comparison: sequential vs parallel
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# Use AsyncOpenAI for async operations
async_client = AsyncOpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


async def async_chat(prompt: str, request_id: int = 0) -> dict:
    """
    Make an async chat completion request.

    Args:
        prompt: The user's message.
        request_id: Identifier for tracking parallel requests.

    Returns:
        Dictionary with response and timing information.
    """
    start_time = time.time()

    response = await async_client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": prompt}],
    )

    elapsed = time.time() - start_time

    return {
        "id": request_id,
        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
        "response": response.choices[0].message.content,
        "time": elapsed,
    }


async def async_stream_chat(prompt: str) -> str:
    """
    Make an async streaming chat request.

    Args:
        prompt: The user's message.

    Returns:
        The complete response text.
    """
    stream = await async_client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_response = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            console.print(chunk.choices[0].delta.content, end="")

    console.print()
    return full_response


async def parallel_requests(prompts: list[str]) -> list[dict]:
    """
    Make multiple requests in parallel using asyncio.gather.

    Args:
        prompts: List of prompts to send concurrently.

    Returns:
        List of response dictionaries.
    """
    tasks = [async_chat(prompt, request_id=i) for i, prompt in enumerate(prompts)]

    results = await asyncio.gather(*tasks)
    return results


async def sequential_requests(prompts: list[str]) -> list[dict]:
    """
    Make multiple requests sequentially for comparison.

    Args:
        prompts: List of prompts to send one at a time.

    Returns:
        List of response dictionaries.
    """
    results = []
    for i, prompt in enumerate(prompts):
        result = await async_chat(prompt, request_id=i)
        results.append(result)
    return results


async def main():
    console.print(
        Panel.fit(
            "[bold blue]Async/Parallel Requests Example[/bold blue]\n"
            "Improve throughput with concurrent API calls",
            border_style="blue",
        )
    )

    # Example 1: Simple async request
    console.print("\n[bold yellow]Example 1: Single Async Request[/bold yellow]")

    result = await async_chat("What is 2 + 2? Answer briefly.")
    console.print(f"[bold green]Prompt:[/bold green] {result['prompt']}")
    console.print(f"[bold cyan]Response:[/bold cyan] {result['response']}")
    console.print(f"[dim]Time: {result['time']:.2f}s[/dim]")

    # Example 2: Parallel requests vs sequential
    console.print("\n[bold yellow]Example 2: Parallel vs Sequential Performance[/bold yellow]")

    prompts = [
        "What color is the sky? One word answer.",
        "What is the capital of Japan? One word answer.",
        "How many legs does a spider have? One word answer.",
        "What planet is known as the Red Planet? One word answer.",
        "What is H2O commonly called? One word answer.",
    ]

    # Sequential execution
    console.print("\n[dim]Running sequential requests...[/dim]")
    seq_start = time.time()
    seq_results = await sequential_requests(prompts)
    seq_time = time.time() - seq_start

    # Parallel execution
    console.print("[dim]Running parallel requests...[/dim]")
    par_start = time.time()
    par_results = await parallel_requests(prompts)
    par_time = time.time() - par_start

    # Results table
    table = Table(title="Performance Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Total Time", style="green")
    table.add_column("Speedup", style="yellow")

    table.add_row("Sequential", f"{seq_time:.2f}s", "1.0x")
    table.add_row("Parallel", f"{par_time:.2f}s", f"{seq_time / par_time:.2f}x")

    console.print(table)

    # Show parallel results
    console.print("\n[bold yellow]Parallel Results:[/bold yellow]")
    for result in par_results:
        response_preview = (
            result["response"][:80] + "..." if len(result["response"]) > 80 else result["response"]
        )
        console.print(f"  [{result['id']}] {response_preview} ({result['time']:.2f}s)")

    # Example 3: Async streaming
    console.print("\n[bold yellow]Example 3: Async Streaming[/bold yellow]")
    console.print("[bold green]Prompt:[/bold green] Write a haiku about code.\n")
    console.print("[bold cyan]Streaming response:[/bold cyan] ", end="")

    await async_stream_chat("Write a haiku about code.")

    # Example 4: Batch processing pattern
    console.print("\n[bold yellow]Example 4: Batch Processing Pattern[/bold yellow]")

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    console.print("[dim]Processing batch of questions in parallel...[/dim]\n")

    batch_results = await parallel_requests(questions)

    for result in batch_results:
        console.print(f"[bold green]Q:[/bold green] {result['prompt']}")
        response_lines = result["response"].split("\n")[0]  # First line only
        console.print(f"[bold cyan]A:[/bold cyan] {response_lines[:150]}...")
        console.print()


if __name__ == "__main__":
    asyncio.run(main())
