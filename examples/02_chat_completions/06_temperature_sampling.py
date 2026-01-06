#!/usr/bin/env python3
"""
06_temperature_sampling.py - Temperature and Sampling Parameters

This example demonstrates how temperature and other sampling parameters
affect the model's output. These parameters control the randomness and
creativity of responses.

Key concepts:
- Temperature: Controls randomness (0 = deterministic, higher = more random)
- Top P (nucleus sampling): Limits token selection to top probability mass
- Comparing outputs at different temperature settings
- When to use different temperature values
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


def chat_with_temperature(prompt: str, temperature: float, top_p: float = 1.0) -> str:
    """
    Make a chat request with specific temperature and top_p settings.

    Args:
        prompt: The user's message.
        temperature: Sampling temperature (0.0 to 2.0).
        top_p: Nucleus sampling parameter (0.0 to 1.0).

    Returns:
        The model's response text.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content


def demonstrate_temperature_effect(prompt: str, temperatures: list[float]):
    """
    Show how different temperatures affect the same prompt.

    Args:
        prompt: The prompt to test.
        temperatures: List of temperature values to try.
    """
    console.print(f"\n[bold green]Prompt:[/bold green] {prompt}\n")

    for temp in temperatures:
        response = chat_with_temperature(prompt, temperature=temp)
        console.print(
            Panel(
                response,
                title=f"[bold]Temperature = {temp}[/bold]",
                border_style=("cyan" if temp < 1.0 else "yellow" if temp < 1.5 else "red"),
            )
        )


def demonstrate_consistency(prompt: str, temperature: float, runs: int = 3):
    """
    Show how temperature affects response consistency.

    Args:
        prompt: The prompt to test.
        temperature: Temperature setting.
        runs: Number of times to run the same prompt.
    """
    console.print(f"\n[bold yellow]Testing consistency at temperature={temperature}[/bold yellow]")
    console.print(f"[dim]Running the same prompt {runs} times...[/dim]\n")

    responses = []
    for i in range(runs):
        response = chat_with_temperature(prompt, temperature=temperature)
        responses.append(response)
        console.print(f"[bold]Run {i + 1}:[/bold] {response[:100]}...")

    # Check if all responses are identical
    unique_responses = len(set(responses))
    if unique_responses == 1:
        console.print(f"\n[green]All {runs} responses were identical (deterministic)[/green]")
    else:
        console.print(f"\n[yellow]{unique_responses} unique responses out of {runs} runs[/yellow]")


def main():
    console.print(
        Panel.fit(
            "[bold blue]Temperature and Sampling Parameters[/bold blue]\n"
            "Control creativity and randomness in responses",
            border_style="blue",
        )
    )

    # Explain temperature settings
    table = Table(title="Temperature Guide")
    table.add_column("Temperature", style="cyan")
    table.add_column("Behavior", style="white")
    table.add_column("Best For", style="green")

    table.add_row("0.0", "Deterministic, most likely tokens", "Factual Q&A, coding, math")
    table.add_row("0.3-0.5", "Low randomness, focused", "Business writing, summaries")
    table.add_row("0.7-0.9", "Balanced creativity", "General conversation, explanations")
    table.add_row("1.0-1.2", "Higher creativity", "Creative writing, brainstorming")
    table.add_row("1.5-2.0", "Very random, experimental", "Wild ideas, poetry, exploration")

    console.print(table)

    # Example 1: Factual question at different temperatures
    console.print("\n[bold yellow]Example 1: Factual Question[/bold yellow]")
    demonstrate_temperature_effect(
        "What is the chemical formula for water? Answer in one word.",
        temperatures=[0.0, 0.5, 1.0, 1.5],
    )

    # Example 2: Creative task at different temperatures
    console.print("\n[bold yellow]Example 2: Creative Task[/bold yellow]")
    demonstrate_temperature_effect(
        "Give me one creative name for a coffee shop.",
        temperatures=[0.0, 0.7, 1.2, 1.8],
    )

    # Example 3: Consistency demonstration
    console.print("\n[bold yellow]Example 3: Consistency Test[/bold yellow]")

    test_prompt = "Name a color. One word only."

    demonstrate_consistency(test_prompt, temperature=0.0, runs=3)
    demonstrate_consistency(test_prompt, temperature=1.0, runs=3)

    # Example 4: Top P (Nucleus Sampling)
    console.print("\n[bold yellow]Example 4: Top P (Nucleus Sampling)[/bold yellow]")
    console.print(
        "[dim]Top P limits token selection to the smallest set whose cumulative probability exceeds P[/dim]\n"
    )

    creative_prompt = "Complete this sentence creatively: The robot decided to..."

    top_p_values = [0.1, 0.5, 0.9, 1.0]

    for top_p in top_p_values:
        response = chat_with_temperature(creative_prompt, temperature=1.0, top_p=top_p)
        console.print(f"[bold]Top P = {top_p}:[/bold] {response[:100]}...")

    # Example 5: Recommended settings by use case
    console.print("\n[bold yellow]Example 5: Recommended Settings by Use Case[/bold yellow]")

    use_cases = [
        ("Code generation", "Write a Python function to reverse a string.", 0.0),
        ("Email writing", "Write a brief professional email declining a meeting.", 0.3),
        ("Story starter", "Write the opening line of a mystery novel.", 1.0),
    ]

    for use_case, prompt, temp in use_cases:
        response = chat_with_temperature(prompt, temperature=temp)
        console.print(
            Panel(
                f"[dim]Prompt: {prompt}[/dim]\n\n{response}",
                title=f"[bold]{use_case}[/bold] (temp={temp})",
                border_style="blue",
            )
        )


if __name__ == "__main__":
    main()
