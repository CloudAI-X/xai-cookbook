#!/usr/bin/env python3
"""
07_max_tokens.py - Controlling max_tokens and Stop Sequences

This example demonstrates how to control the length of model responses
using max_tokens and stop sequences. These parameters are essential for
managing costs, response format, and output structure.

Key concepts:
- max_tokens: Limits the maximum number of tokens in the response
- stop sequences: Custom strings that cause generation to stop
- Handling truncated responses
- Token counting and cost estimation
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from xai_cookbook.client import DEFAULT_MODEL, DEFAULT_NON_REASONING_MODEL

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


def chat_with_max_tokens(prompt: str, max_tokens: int, stop: list[str] | None = None) -> dict:
    """
    Make a chat request with max_tokens and optional stop sequences.

    Args:
        prompt: The user's message.
        max_tokens: Maximum tokens in the response.
        stop: Optional list of stop sequences.

    Returns:
        Dictionary with response and metadata.
    """
    # Note: stop sequences are not supported by reasoning models
    # Use non-reasoning model when stop sequences are needed
    model = DEFAULT_NON_REASONING_MODEL if stop else DEFAULT_MODEL
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    if stop:
        kwargs["stop"] = stop

    response = client.chat.completions.create(**kwargs)

    return {
        "content": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }


def demonstrate_max_tokens_effect(prompt: str, token_limits: list[int]):
    """
    Show how different max_tokens values affect response length.

    Args:
        prompt: The prompt to test.
        token_limits: List of max_tokens values to try.
    """
    console.print(f"\n[bold green]Prompt:[/bold green] {prompt}\n")

    for max_tokens in token_limits:
        result = chat_with_max_tokens(prompt, max_tokens=max_tokens)

        # Determine if response was truncated
        truncated = result["finish_reason"] == "length"
        status = "[red](truncated)[/red]" if truncated else "[green](complete)[/green]"

        console.print(
            Panel(
                f"{result['content']}\n\n"
                f"[dim]Tokens used: {result['completion_tokens']} | "
                f"Finish reason: {result['finish_reason']}[/dim]",
                title=f"[bold]max_tokens = {max_tokens}[/bold] {status}",
                border_style="red" if truncated else "green",
            )
        )


def main():
    console.print(
        Panel.fit(
            "[bold blue]Max Tokens and Stop Sequences[/bold blue]\n"
            "Control response length and termination",
            border_style="blue",
        )
    )

    # Example 1: max_tokens comparison
    console.print("\n[bold yellow]Example 1: Effect of max_tokens on Response Length[/bold yellow]")

    demonstrate_max_tokens_effect(
        "Explain the theory of relativity.", token_limits=[20, 50, 100, 200]
    )

    # Example 2: Stop sequences
    console.print("\n[bold yellow]Example 2: Stop Sequences[/bold yellow]")
    console.print("[dim]Stop sequences cause generation to halt when encountered[/dim]\n")

    # Example: Stop at newline
    prompt1 = "List three fruits:\n1."
    result1 = chat_with_max_tokens(
        prompt1,
        max_tokens=100,
        stop=["\n2."],  # Stop before second item
    )
    console.print(f"[bold green]Prompt:[/bold green] {prompt1}")
    console.print(f"[bold cyan]Response (stops at '\\n2.'):[/bold cyan] {result1['content']}")
    console.print(f"[dim]Finish reason: {result1['finish_reason']}[/dim]\n")

    # Example: Stop at punctuation
    prompt2 = "Tell me a fun fact about dolphins."
    result2 = chat_with_max_tokens(
        prompt2,
        max_tokens=200,
        stop=[".", "!", "?"],  # Stop at first sentence-ending punctuation
    )
    console.print(f"[bold green]Prompt:[/bold green] {prompt2}")
    console.print(f"[bold cyan]Response (stops at sentence end):[/bold cyan] {result2['content']}")
    console.print(f"[dim]Finish reason: {result2['finish_reason']}[/dim]\n")

    # Example: Multiple stop sequences for structured output
    prompt3 = "Generate a JSON object with name and age fields for a person named Alice."
    result3 = chat_with_max_tokens(
        prompt3,
        max_tokens=100,
        stop=["\n\n", "```"],  # Stop at double newline or code block end
    )
    console.print(f"[bold green]Prompt:[/bold green] {prompt3}")
    console.print(f"[bold cyan]Response:[/bold cyan]\n{result3['content']}")
    console.print(f"[dim]Finish reason: {result3['finish_reason']}[/dim]")

    # Example 3: Token usage tracking
    console.print("\n[bold yellow]Example 3: Token Usage Tracking[/bold yellow]")

    prompts_for_tracking = [
        ("Short question", "What is 2+2?"),
        ("Medium question", "Explain what machine learning is in one paragraph."),
        (
            "Long question",
            "Write a detailed explanation of how neural networks work, including input layers, hidden layers, output layers, and backpropagation.",
        ),
    ]

    table = Table(title="Token Usage by Prompt Length")
    table.add_column("Prompt Type", style="cyan")
    table.add_column("Prompt Tokens", style="yellow")
    table.add_column("Completion Tokens", style="green")
    table.add_column("Total Tokens", style="magenta")

    for prompt_type, prompt in prompts_for_tracking:
        result = chat_with_max_tokens(prompt, max_tokens=150)
        table.add_row(
            prompt_type,
            str(result["prompt_tokens"]),
            str(result["completion_tokens"]),
            str(result["total_tokens"]),
        )

    console.print(table)

    # Example 4: Handling truncated responses
    console.print("\n[bold yellow]Example 4: Detecting and Handling Truncation[/bold yellow]")

    long_prompt = "Write a detailed 500-word essay about climate change."

    # First, try with low max_tokens
    result_truncated = chat_with_max_tokens(long_prompt, max_tokens=50)

    if result_truncated["finish_reason"] == "length":
        console.print("[yellow]Warning: Response was truncated![/yellow]")
        console.print(f"[dim]Response preview: {result_truncated['content'][:100]}...[/dim]\n")

        # Retry with higher limit
        console.print("[dim]Retrying with higher max_tokens...[/dim]")
        result_full = chat_with_max_tokens(long_prompt, max_tokens=500)

        console.print("[green]Full response received[/green]")
        console.print(f"[dim]Tokens used: {result_full['completion_tokens']}[/dim]")
        console.print(f"[dim]Finish reason: {result_full['finish_reason']}[/dim]")

    # Example 5: Best practices summary
    console.print("\n[bold yellow]Example 5: Best Practices Summary[/bold yellow]")

    best_practices = Table(title="max_tokens Best Practices")
    best_practices.add_column("Use Case", style="cyan")
    best_practices.add_column("Recommended max_tokens", style="green")
    best_practices.add_column("Notes", style="white")

    best_practices.add_row("Quick Q&A", "50-100", "Short, direct answers")
    best_practices.add_row("Summaries", "150-300", "Paragraph-length responses")
    best_practices.add_row("Explanations", "300-500", "Detailed explanations")
    best_practices.add_row("Code generation", "500-1000", "Complete code snippets")
    best_practices.add_row("Long-form content", "1000-4000", "Essays, articles, documentation")

    console.print(best_practices)


if __name__ == "__main__":
    main()
