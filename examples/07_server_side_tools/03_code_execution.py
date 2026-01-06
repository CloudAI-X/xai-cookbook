#!/usr/bin/env python3
"""
03_code_execution.py - Code Execution Server-Side Tool

This example demonstrates how to use xAI's code execution capability.
This server-side tool allows Grok to write and execute Python code
to solve problems, perform calculations, and process data.

Code execution is useful for:
- Mathematical calculations
- Data analysis and visualization
- Algorithm implementation
- Problem solving that requires computation

Key concepts:
- Enabling code execution through extra_body
- Understanding code execution capabilities
- Handling code execution results
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

console = Console()


def get_client() -> OpenAI:
    """Initialize and return the xAI client."""
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] X_AI_API_KEY environment variable not set.\n"
            "Please set it in your .env file or environment."
        )
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def execute_code_task(client: OpenAI, task: str) -> str:
    """
    Ask Grok to solve a problem using code execution.

    Grok will write and execute Python code server-side to solve
    the given task.

    Args:
        client: The OpenAI client configured for xAI.
        task: Description of the task to solve with code.

    Returns:
        The assistant's response including code execution results.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can write and execute Python code to solve problems. When appropriate, write code to calculate or process data accurately.",
            },
            {"role": "user", "content": task},
        ],
    )

    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Code Execution Server-Side Tool[/bold blue]\n"
            "Execute Python code for calculations and data processing",
            border_style="blue",
        )
    )

    client = get_client()

    # Example 1: Mathematical calculation
    console.print("\n[bold cyan]Example 1: Mathematical Calculation[/bold cyan]")
    console.print(
        "[dim]Task: 'Calculate the compound interest on $10,000 at 5% annual "
        "rate for 10 years, compounded monthly'[/dim]"
    )

    task1 = (
        "Calculate the compound interest on $10,000 at 5% annual rate "
        "for 10 years, compounded monthly. Show the formula and the result."
    )
    response1 = execute_code_task(client, task1)

    console.print(Panel(Markdown(response1), title="Calculation Result", border_style="green"))

    # Example 2: Data analysis
    console.print("\n[bold cyan]Example 2: Data Analysis[/bold cyan]")
    console.print(
        "[dim]Task: 'Find the mean, median, and standard deviation of these "
        "numbers: 12, 15, 18, 22, 25, 28, 31, 35, 38, 42'[/dim]"
    )

    task2 = (
        "Calculate the mean, median, and standard deviation of these numbers: "
        "12, 15, 18, 22, 25, 28, 31, 35, 38, 42. Show your calculations."
    )
    response2 = execute_code_task(client, task2)

    console.print(Panel(Markdown(response2), title="Analysis Result", border_style="green"))

    # Example 3: Algorithm implementation
    console.print("\n[bold cyan]Example 3: Algorithm Problem[/bold cyan]")
    console.print(
        "[dim]Task: 'Find all prime numbers between 1 and 100 and calculate their sum'[/dim]"
    )

    task3 = "Find all prime numbers between 1 and 100 and calculate their sum."
    response3 = execute_code_task(client, task3)

    console.print(Panel(Markdown(response3), title="Algorithm Result", border_style="green"))

    # Example 4: String manipulation
    console.print("\n[bold cyan]Example 4: String Processing[/bold cyan]")
    console.print(
        "[dim]Task: 'Analyze the word frequency in this sentence and find the "
        "most common words'[/dim]"
    )

    task4 = (
        "Analyze this text and count word frequency: "
        "'The quick brown fox jumps over the lazy dog. The dog sleeps while "
        "the fox runs. The brown fox is quick.' "
        "Show the top 5 most frequent words."
    )
    response4 = execute_code_task(client, task4)

    console.print(Panel(Markdown(response4), title="Text Analysis Result", border_style="green"))

    # Show code execution capabilities
    console.print("\n[bold yellow]Code Execution Capabilities:[/bold yellow]")
    console.print(
        """
    [cyan]Mathematical Operations[/cyan]
      - Complex calculations and formulas
      - Statistical analysis
      - Financial calculations

    [cyan]Data Processing[/cyan]
      - List and string manipulation
      - Data transformation
      - Pattern matching

    [cyan]Algorithm Implementation[/cyan]
      - Sorting and searching
      - Graph algorithms
      - Dynamic programming

    [cyan]Limitations[/cyan]
      - No external library imports (beyond standard library)
      - No file system access
      - No network requests
      - Execution time limits
    """
    )


if __name__ == "__main__":
    main()
