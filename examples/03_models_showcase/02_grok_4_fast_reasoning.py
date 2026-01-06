#!/usr/bin/env python3
"""
Grok-4 Fast Reasoning - Extended Context with Reasoning Mode

Model: grok-4-fast-reasoning
Context Window: 2M tokens (massive!)
Mode: Reasoning-enabled for step-by-step thinking

Grok-4 Fast Reasoning excels at:
- Processing extremely long documents
- Step-by-step logical reasoning
- Complex mathematical problems
- Multi-document analysis
- Research and synthesis tasks

Best for: Long-context tasks requiring explicit reasoning
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

console = Console()


def create_client() -> OpenAI:
    """Create xAI client using OpenAI SDK."""
    return OpenAI(api_key=os.environ["X_AI_API_KEY"], base_url="https://api.x.ai/v1")


def demonstrate_reasoning_mode(client: OpenAI) -> None:
    """Demonstrate the reasoning capabilities with step-by-step thinking."""
    console.print(
        Panel.fit(
            "[bold cyan]Grok-4 Fast Reasoning Demo[/bold cyan]\n"
            "Model: grok-4-fast-reasoning | Context: 2M tokens | Reasoning Mode",
            border_style="cyan",
        )
    )

    # Mathematical reasoning problem
    prompt = """Solve this problem step by step:

A farmer has a rectangular field that is 120 meters long. He wants to divide it
into three sections using two fences parallel to the width. The first section
should be twice as long as the second, and the third should be 20 meters longer
than the second.

What is the length of each section? Also calculate the percentage of the total
field each section represents."""

    console.print("\n[bold]Task:[/bold] Mathematical Reasoning Problem\n")
    console.print(Panel(prompt, title="Problem", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a mathematics tutor. Show your reasoning "
                "step by step, explaining each calculation clearly.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # Lower temperature for precise reasoning
        max_tokens=1500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4 Fast Reasoning Solution:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))

    if response.usage:
        console.print(
            f"\n[dim]Tokens used - Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}[/dim]"
        )


def demonstrate_long_context_analysis(client: OpenAI) -> None:
    """Demonstrate ability to handle complex multi-part analysis."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Complex Logical Reasoning\n")

    # Logic puzzle requiring careful reasoning
    prompt = """Solve this logic puzzle:

Five houses in a row are painted different colors: red, green, blue, yellow, white.
Five people of different nationalities live in these houses: American, British,
Canadian, Danish, Egyptian.

Clues:
1. The British person lives in the red house.
2. The American lives immediately to the right of the blue house.
3. The green house is immediately to the left of the white house.
4. The Danish person lives in the yellow house.
5. The Canadian does not live in the house on either end.
6. The Egyptian lives in the first house (leftmost).
7. The green house is not on either end.

Determine who lives in which colored house, and the order from left to right.
Show your reasoning for each deduction."""

    console.print(Panel(prompt, title="Logic Puzzle", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a logic puzzle expert. Solve puzzles "
                "methodically, showing each logical deduction.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4 Fast Reasoning Solution:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_code_reasoning(client: OpenAI) -> None:
    """Demonstrate reasoning about code correctness."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Code Correctness Reasoning\n")

    prompt = """Analyze this Python function for correctness and efficiency:

```python
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates
```

1. Is this function correct? Prove it or find a counterexample.
2. What is the time complexity? Explain your analysis.
3. Suggest an optimized version and explain why it's better."""

    console.print(Panel(prompt, title="Code Analysis", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a computer science professor. Analyze code "
                "rigorously with formal reasoning.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4 Fast Reasoning Analysis:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Grok-4 Fast Reasoning Demonstration[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    demonstrate_reasoning_mode(client)
    demonstrate_long_context_analysis(client)
    demonstrate_code_reasoning(client)

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print("[dim]Grok-4 Fast Reasoning excels at step-by-step problem solving.[/dim]\n")
