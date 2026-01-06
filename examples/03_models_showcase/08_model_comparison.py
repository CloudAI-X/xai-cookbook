#!/usr/bin/env python3
"""
Model Comparison - Compare Multiple xAI Models

This script compares different xAI models on the same task
to help you understand their relative strengths and choose
the right model for your use case.

Models compared:
- grok-4-1-fast-reasoning (Latest flagship - reasoning)
- grok-4-1-fast-non-reasoning (Latest flagship - speed)
- grok-4-fast-reasoning (Previous gen - reasoning)
- grok-4-0709 (Premium flagship)
- grok-3 (General purpose)
- grok-3-mini (Cost-effective)
- grok-code-fast-1 (Code-specialized)
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv()

console = Console()

# Models to compare with their characteristics
MODELS = {
    "grok-4-1-fast-reasoning": {
        "name": "Grok-4.1 Fast Reasoning",
        "context": "2M",
        "input_cost": "$0.20",
        "output_cost": "$0.50",
        "strength": "Latest SOTA + Reasoning",
    },
    "grok-4-1-fast-non-reasoning": {
        "name": "Grok-4.1 Fast",
        "context": "2M",
        "input_cost": "$0.20",
        "output_cost": "$0.50",
        "strength": "Latest SOTA + Speed",
    },
    "grok-4-fast-reasoning": {
        "name": "Grok-4 Fast Reasoning",
        "context": "2M",
        "input_cost": "$0.20",
        "output_cost": "$0.50",
        "strength": "Reasoning mode",
    },
    "grok-4-fast-non-reasoning": {
        "name": "Grok-4 Fast",
        "context": "2M",
        "input_cost": "$0.20",
        "output_cost": "$0.50",
        "strength": "High throughput",
    },
    "grok-4-0709": {
        "name": "Grok-4 Premium",
        "context": "256K",
        "input_cost": "$3.00",
        "output_cost": "$15.00",
        "strength": "Maximum capability",
    },
    "grok-3": {
        "name": "Grok-3",
        "context": "131K",
        "input_cost": "$3.00",
        "output_cost": "$15.00",
        "strength": "General purpose",
    },
    "grok-3-mini": {
        "name": "Grok-3 Mini",
        "context": "131K",
        "input_cost": "$0.30",
        "output_cost": "$0.50",
        "strength": "Budget-friendly",
    },
    "grok-code-fast-1": {
        "name": "Grok Code Fast",
        "context": "256K",
        "input_cost": "$0.20",
        "output_cost": "$1.50",
        "strength": "Code specialized",
    },
}


def create_client() -> OpenAI:
    """Create xAI client using OpenAI SDK."""
    return OpenAI(api_key=os.environ["X_AI_API_KEY"], base_url="https://api.x.ai/v1")


def display_model_info() -> None:
    """Display information about available models."""
    table = Table(title="xAI Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Context", justify="center")
    table.add_column("Input/1M", justify="right")
    table.add_column("Output/1M", justify="right")
    table.add_column("Best For", style="yellow")

    for model_id, info in MODELS.items():
        table.add_row(
            info["name"],
            info["context"],
            info["input_cost"],
            info["output_cost"],
            info["strength"],
        )

    console.print(table)
    console.print()


def compare_on_task(
    client: OpenAI,
    task_name: str,
    system_prompt: str,
    user_prompt: str,
    models: list[str],
) -> dict[str, tuple[str, float, int, int]]:
    """
    Compare multiple models on the same task.

    Returns dict of model_id -> (response, time, input_tokens, output_tokens)
    """
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for model_id in models:
            task = progress.add_task(f"Running {MODELS[model_id]['name']}...", total=None)

            start_time = time.time()

            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                )

                elapsed = time.time() - start_time
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0

                results[model_id] = (content, elapsed, input_tokens, output_tokens)

            except Exception as e:
                results[model_id] = (f"Error: {str(e)}", 0, 0, 0)

            progress.remove_task(task)

    return results


def display_comparison_results(
    task_name: str, results: dict[str, tuple[str, float, int, int]]
) -> None:
    """Display comparison results in a readable format."""
    console.print(f"\n[bold cyan]Results for: {task_name}[/bold cyan]\n")

    # Performance table
    perf_table = Table(title="Performance Metrics")
    perf_table.add_column("Model", style="cyan")
    perf_table.add_column("Response Time", justify="right")
    perf_table.add_column("Input Tokens", justify="right")
    perf_table.add_column("Output Tokens", justify="right")

    for model_id, (content, elapsed, input_tok, output_tok) in results.items():
        perf_table.add_row(
            MODELS[model_id]["name"], f"{elapsed:.2f}s", str(input_tok), str(output_tok)
        )

    console.print(perf_table)
    console.print()

    # Response content
    for model_id, (content, elapsed, _, _) in results.items():
        console.print(
            Panel(
                Markdown(content[:1500] + ("..." if len(content) > 1500 else "")),
                title=f"{MODELS[model_id]['name']} Response",
                border_style="green" if "Error" not in content else "red",
            )
        )
        console.print()


def run_reasoning_comparison(client: OpenAI) -> None:
    """Compare models on a reasoning task - showcases reasoning models."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Comparison 1: Reasoning Task (Flagship Focus)[/bold]")
    console.print("=" * 60 + "\n")

    task_prompt = """A bat and a ball cost $1.10 in total. The bat costs $1.00 more than
the ball. How much does the ball cost? Show your reasoning step by step."""

    # Compare the flagship models on reasoning
    results = compare_on_task(
        client,
        "Reasoning Task",
        "You are a helpful assistant. Show your reasoning clearly.",
        task_prompt,
        ["grok-4-1-fast-reasoning", "grok-4-fast-reasoning", "grok-4-0709"],
    )

    console.print(Panel(task_prompt, title="Task", border_style="yellow"))
    display_comparison_results("Reasoning Task", results)


def run_speed_comparison(client: OpenAI) -> None:
    """Compare non-reasoning models for speed."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Comparison 2: Speed Task (Non-Reasoning Models)[/bold]")
    console.print("=" * 60 + "\n")

    task_prompt = """Summarize the key benefits of cloud computing in 3 bullet points."""

    results = compare_on_task(
        client,
        "Speed Task",
        "You are a concise assistant. Be brief and direct.",
        task_prompt,
        ["grok-4-1-fast-non-reasoning", "grok-4-fast-non-reasoning", "grok-3-mini"],
    )

    console.print(Panel(task_prompt, title="Task", border_style="yellow"))
    display_comparison_results("Speed Task", results)


def run_code_comparison(client: OpenAI) -> None:
    """Compare models on a coding task."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Comparison 3: Coding Task[/bold]")
    console.print("=" * 60 + "\n")

    task_prompt = """Write a Python function that finds the longest palindromic
substring in a given string. Include time complexity analysis."""

    results = compare_on_task(
        client,
        "Coding Task",
        "You are an expert programmer. Write clean, efficient code.",
        task_prompt,
        ["grok-4-1-fast-reasoning", "grok-code-fast-1", "grok-3"],
    )

    console.print(Panel(task_prompt, title="Task", border_style="yellow"))
    display_comparison_results("Coding Task", results)


def run_creative_comparison(client: OpenAI) -> None:
    """Compare models on a creative task."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Comparison 4: Creative Task[/bold]")
    console.print("=" * 60 + "\n")

    task_prompt = """Write a haiku about artificial intelligence. Then explain the
symbolism and imagery you used."""

    results = compare_on_task(
        client,
        "Creative Task",
        "You are a creative writer with expertise in poetry.",
        task_prompt,
        ["grok-4-1-fast-reasoning", "grok-4-0709", "grok-3"],
    )

    console.print(Panel(task_prompt, title="Task", border_style="yellow"))
    display_comparison_results("Creative Task", results)


def run_cost_comparison(client: OpenAI) -> None:
    """Compare cost-effective models."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Comparison 5: Cost-Effective Models[/bold]")
    console.print("=" * 60 + "\n")

    task_prompt = """Classify these reviews as POSITIVE, NEGATIVE, or NEUTRAL:

1. "Best product I've ever bought! Highly recommend!"
2. "It's okay, nothing special."
3. "Terrible quality, complete waste of money."
4. "Works as expected, delivery was fast."
5. "Exceeded my expectations in every way!"

Return just the classification for each."""

    results = compare_on_task(
        client,
        "Classification Task",
        "You are a sentiment classifier. Be concise.",
        task_prompt,
        ["grok-4-1-fast-non-reasoning", "grok-3-mini", "grok-3"],
    )

    console.print(Panel(task_prompt, title="Task", border_style="yellow"))
    display_comparison_results("Classification Task", results)


def print_recommendations() -> None:
    """Print model selection recommendations."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Model Selection Guide[/bold cyan]")
    console.print("=" * 60 + "\n")

    recommendations = """
## When to Use Each Model

### Grok-4.1 Fast Reasoning (grok-4-1-fast-reasoning) - RECOMMENDED DEFAULT
- **Latest and most capable model**
- Complex reasoning and analysis
- Step-by-step problem solving
- Mathematical proofs and logic
- 2M token context window
- Best quality-to-cost ratio

### Grok-4.1 Fast Non-Reasoning (grok-4-1-fast-non-reasoning)
- High-throughput applications
- Quick summarization
- Information extraction
- When speed matters more than deep reasoning
- 2M token context window

### Grok-4 Fast Reasoning (grok-4-fast-reasoning)
- Previous generation reasoning model
- Still excellent for complex tasks
- 2M token context window

### Grok-4 Fast Non-Reasoning (grok-4-fast-non-reasoning)
- Previous generation speed model
- High-volume processing
- 2M token context window

### Grok-4 Premium (grok-4-0709)
- Maximum capability when cost is no concern
- Ultra-high-stakes decisions
- Complex nuanced analysis
- 256K context window

### Grok-3 (grok-3)
- General-purpose tasks
- Standard business applications
- Content generation
- 131K context window

### Grok-3 Mini (grok-3-mini)
- Budget-sensitive applications
- High-volume simple tasks
- Prototyping and development
- 131K context window

### Grok Code Fast (grok-code-fast-1)
- Code generation and review
- Bug detection
- Test generation
- Any code-focused task
- 256K context window
"""

    console.print(Panel(Markdown(recommendations), border_style="cyan"))


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Model Comparison Tool[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    # Display model information
    display_model_info()

    # Run comparisons
    console.print("[bold]Running model comparisons...[/bold]\n")

    run_reasoning_comparison(client)
    run_speed_comparison(client)
    run_code_comparison(client)
    run_creative_comparison(client)
    run_cost_comparison(client)

    # Print recommendations
    print_recommendations()

    console.print("\n[bold cyan]Comparison complete![/bold cyan]")
    console.print("[dim]Use these insights to choose the right model for your needs.[/dim]\n")
