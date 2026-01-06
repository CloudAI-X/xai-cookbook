#!/usr/bin/env python3
"""
04_streaming.py - Streaming Responses in Real-Time

This example demonstrates how to stream responses from the xAI API.
Streaming allows you to display tokens as they're generated, providing
a more responsive user experience.

Key concepts:
- Enabling streaming with stream=True
- Processing stream chunks
- Building up response incrementally
- Handling stream events
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


def stream_chat(prompt: str, system_prompt: str | None = None) -> str:
    """
    Stream a chat response and print tokens as they arrive.

    Args:
        prompt: The user's message.
        system_prompt: Optional system instructions.

    Returns:
        The complete response text.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Create streaming response
    stream = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=messages,
        stream=True,  # Enable streaming
    )

    full_response = ""

    # Process each chunk as it arrives
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token
            # Print token without newline to show streaming effect
            console.print(token, end="")

    # Print newline at the end
    console.print()

    return full_response


def stream_with_live_display(prompt: str) -> str:
    """
    Stream a response using Rich's Live display for better formatting.

    Args:
        prompt: The user's message.

    Returns:
        The complete response text.
    """
    messages = [{"role": "user", "content": prompt}]

    stream = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=messages,
        stream=True,
    )

    full_response = ""

    # Use Rich Live display for smooth updates
    with Live(console=console, refresh_per_second=10) as live:
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                # Update the live display with current text
                live.update(
                    Panel(
                        Text(full_response, style="cyan"),
                        title="[bold]Streaming Response[/bold]",
                        border_style="blue",
                    )
                )

    return full_response


def measure_streaming_performance(prompt: str) -> dict:
    """
    Measure time-to-first-token and total streaming time.

    Args:
        prompt: The user's message.

    Returns:
        Dictionary with timing metrics.
    """
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    first_token_time = None
    token_count = 0

    stream = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1

    end_time = time.time()

    return {
        "time_to_first_token": first_token_time - start_time if first_token_time else 0,
        "total_time": end_time - start_time,
        "token_count": token_count,
        "tokens_per_second": (token_count / (end_time - start_time) if token_count > 0 else 0),
    }


def main():
    console.print(
        Panel.fit(
            "[bold blue]Streaming Responses Example[/bold blue]\nWatch tokens appear in real-time",
            border_style="blue",
        )
    )

    # Example 1: Basic streaming
    console.print("\n[bold yellow]Example 1: Basic Streaming[/bold yellow]")
    console.print("[bold green]User:[/bold green] Write a short poem about AI.\n")
    console.print("[bold cyan]Grok (streaming):[/bold cyan] ", end="")

    stream_chat("Write a short poem about AI (4 lines).")

    # Example 2: Streaming with system prompt
    console.print("\n[bold yellow]Example 2: Streaming with System Prompt[/bold yellow]")
    console.print("[bold green]User:[/bold green] Explain quantum computing.\n")
    console.print("[bold cyan]Grok (streaming):[/bold cyan] ", end="")

    stream_chat(
        "Explain quantum computing in simple terms.",
        system_prompt="You are a concise explainer. Use 2-3 sentences maximum.",
    )

    # Example 3: Live display streaming
    console.print("\n[bold yellow]Example 3: Rich Live Display Streaming[/bold yellow]")
    console.print("[bold green]User:[/bold green] Tell me 3 fun facts about space.\n")

    stream_with_live_display("Tell me 3 fun facts about space. Keep it brief.")

    # Example 4: Performance metrics
    console.print("\n[bold yellow]Example 4: Streaming Performance Metrics[/bold yellow]")

    metrics = measure_streaming_performance("Count from 1 to 10 with brief descriptions.")

    console.print(f"  Time to first token: {metrics['time_to_first_token']:.3f}s")
    console.print(f"  Total streaming time: {metrics['total_time']:.3f}s")
    console.print(f"  Total chunks received: {metrics['token_count']}")
    console.print(f"  Chunks per second: {metrics['tokens_per_second']:.1f}")


if __name__ == "__main__":
    main()
