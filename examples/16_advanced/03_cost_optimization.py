#!/usr/bin/env python3
"""
03_cost_optimization.py - Tips for Reducing Costs

This example provides strategies and best practices for optimizing
costs when using the xAI API, including model selection, token
management, and efficient request patterns.

Key concepts:
- Model selection for cost efficiency
- Token optimization techniques
- Caching strategies
- Batch processing
- Monitoring and budgeting
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("X_AI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

console = Console()


@dataclass
class ModelPricing:
    """Pricing information for xAI models."""

    model: str
    input_per_m: float  # $ per million input tokens
    output_per_m: float  # $ per million output tokens
    context_window: int
    best_for: str


# Model pricing (approximate - check docs.x.ai for current pricing)
MODELS = [
    ModelPricing("grok-4-1-fast-reasoning", 0.30, 0.50, 131072, "Simple tasks, high volume"),
    ModelPricing("grok-3", 3.00, 15.00, 131072, "Complex reasoning"),
    ModelPricing("grok-4", 2.00, 10.00, 131072, "Most capable"),
    ModelPricing("grok-4-fast", 5.00, 25.00, 131072, "Fast complex tasks"),
]


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate estimated cost for a request."""
    pricing = next((m for m in MODELS if m.model == model), MODELS[0])

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_m
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_m

    return input_cost + output_cost


def compare_model_costs(
    task_description: str,
    estimated_input: int,
    estimated_output: int,
) -> list[dict]:
    """Compare costs across different models for a task."""
    results = []

    for model in MODELS:
        cost = estimate_cost(model.model, estimated_input, estimated_output)
        results.append(
            {
                "model": model.model,
                "input_cost": (estimated_input / 1_000_000) * model.input_per_m,
                "output_cost": (estimated_output / 1_000_000) * model.output_per_m,
                "total_cost": cost,
                "best_for": model.best_for,
            }
        )

    return sorted(results, key=lambda x: x["total_cost"])


def optimize_prompt(prompt: str) -> str:
    """
    Optimize a prompt for token efficiency.

    Strategies:
    - Remove unnecessary whitespace
    - Use concise language
    - Remove redundant instructions
    """
    # Remove extra whitespace
    lines = [line.strip() for line in prompt.split("\n") if line.strip()]
    return "\n".join(lines)


def main():
    console.print(
        Panel.fit(
            "[bold blue]Cost Optimization Guide[/bold blue]\nStrategies for reducing API costs",
            border_style="blue",
        )
    )

    # Model pricing comparison
    console.print("\n[bold yellow]Model Pricing Comparison:[/bold yellow]")

    pricing_table = Table(show_header=True, header_style="bold cyan")
    pricing_table.add_column("Model", style="green")
    pricing_table.add_column("Input ($/M)", justify="right")
    pricing_table.add_column("Output ($/M)", justify="right")
    pricing_table.add_column("Best For")

    for model in MODELS:
        pricing_table.add_row(
            model.model,
            f"${model.input_per_m:.2f}",
            f"${model.output_per_m:.2f}",
            model.best_for,
        )

    console.print(pricing_table)
    console.print("[dim]Note: Check docs.x.ai for current pricing[/dim]")

    # Cost comparison example
    console.print("\n[bold yellow]Cost Comparison Example:[/bold yellow]")
    console.print("[dim]Task: 1000 tokens input, 500 tokens output[/dim]")

    comparison = compare_model_costs("Example task", 1000, 500)

    cost_table = Table(show_header=True, header_style="bold cyan")
    cost_table.add_column("Model", style="green")
    cost_table.add_column("Input Cost", justify="right")
    cost_table.add_column("Output Cost", justify="right")
    cost_table.add_column("Total", justify="right")
    cost_table.add_column("Savings vs Max", justify="right")

    max_cost = max(c["total_cost"] for c in comparison)

    for c in comparison:
        savings = ((max_cost - c["total_cost"]) / max_cost * 100) if max_cost > 0 else 0
        cost_table.add_row(
            c["model"],
            f"${c['input_cost']:.6f}",
            f"${c['output_cost']:.6f}",
            f"${c['total_cost']:.6f}",
            f"{savings:.0f}%",
        )

    console.print(cost_table)

    # Optimization strategies
    console.print("\n[bold yellow]Optimization Strategy 1: Model Selection[/bold yellow]")
    console.print(
        """
  [cyan]Match model to task complexity:[/cyan]

  - [green]grok-4-1-fast-reasoning:[/green] Simple tasks, classification, Q&A
    - 10x cheaper than grok-3 for input
    - Good for high-volume, low-complexity

  - [green]grok-3:[/green] Complex reasoning, analysis
    - When grok-4-1-fast-reasoning quality is insufficient
    - Balance of cost and capability

  - [green]grok-4:[/green] Most capable tasks
    - When you need the best quality
    - Complex multi-step reasoning

  [dim]Tip: Start with grok-4-1-fast-reasoning, upgrade only if needed[/dim]
"""
    )

    console.print("\n[bold yellow]Optimization Strategy 2: Token Reduction[/bold yellow]")
    console.print(
        """
  [cyan]Minimize input tokens:[/cyan]

  1. [green]Concise prompts:[/green]
     - Remove unnecessary words
     - Use bullet points
     - Avoid repetition

  2. [green]System prompt optimization:[/green]
     - Keep system prompts short
     - Cache static instructions

  3. [green]Conversation pruning:[/green]
     - Summarize long conversations
     - Remove old context

  4. [green]Use max_tokens:[/green]
     - Limit output length
     - Prevent runaway responses
"""
    )

    # Prompt optimization example
    console.print("\n[bold yellow]Prompt Optimization Example:[/bold yellow]")

    verbose_prompt = """
    Hello! I would like you to please help me with something.
    Could you kindly summarize the following text for me?
    I would really appreciate it if you could make it concise.
    Here is the text I need summarized:

    [Your text here]

    Thank you so much for your help!
    """

    optimized_prompt = """Summarize this text concisely:

[Your text here]"""

    console.print("[bold red]Before (verbose):[/bold red]")
    console.print(f"[dim]{verbose_prompt.strip()}[/dim]")
    console.print(f"[dim]~{len(verbose_prompt.split())} words[/dim]")

    console.print("\n[bold green]After (optimized):[/bold green]")
    console.print(f"[dim]{optimized_prompt.strip()}[/dim]")
    console.print(f"[dim]~{len(optimized_prompt.split())} words[/dim]")

    console.print("\n[bold yellow]Optimization Strategy 3: Caching[/bold yellow]")
    console.print(
        """
  [cyan]Reduce duplicate requests:[/cyan]

  1. [green]Response caching:[/green]
     - Cache identical prompts
     - Use TTL for freshness

  2. [green]Embedding caching:[/green]
     - Cache embeddings for repeated text
     - Use vector DB for similarity

  3. [green]Prompt caching:[/green]
     - xAI may cache repeated prompts
     - Check cached_tokens in response

  [dim]Example: Simple cache implementation[/dim]
"""
    )

    console.print(
        """
[dim]from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_chat(prompt_hash: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def chat(prompt: str) -> str:
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return cached_chat(prompt_hash, prompt)[/dim]
"""
    )

    console.print("\n[bold yellow]Optimization Strategy 4: Batch Processing[/bold yellow]")
    console.print(
        """
  [cyan]Process multiple items efficiently:[/cyan]

  1. [green]Batch in single request:[/green]
     - Combine multiple items in one prompt
     - Reduces per-request overhead

  2. [green]Parallel requests:[/green]
     - Process batches concurrently
     - Stay within rate limits

  3. [green]Async processing:[/green]
     - Non-blocking requests
     - Better throughput
"""
    )

    # Monthly cost projection
    console.print("\n[bold yellow]Monthly Cost Projection:[/bold yellow]")

    projection_table = Table(show_header=True, header_style="bold cyan")
    projection_table.add_column("Usage Level", style="green")
    projection_table.add_column("Requests/Day")
    projection_table.add_column("grok-4-1-fast-reasoning")
    projection_table.add_column("grok-3")
    projection_table.add_column("grok-4")

    # Assume 500 input + 200 output tokens per request
    avg_input, avg_output = 500, 200

    scenarios = [
        ("Light", 100),
        ("Medium", 1000),
        ("Heavy", 10000),
        ("Enterprise", 100000),
    ]

    for name, daily_requests in scenarios:
        monthly_requests = daily_requests * 30

        costs = {}
        for model in ["grok-4-1-fast-reasoning", "grok-3", "grok-4"]:
            total_input = monthly_requests * avg_input
            total_output = monthly_requests * avg_output
            cost = estimate_cost(model, total_input, total_output)
            costs[model] = cost

        projection_table.add_row(
            name,
            f"{daily_requests:,}",
            f"${costs['grok-4-1-fast-reasoning']:.2f}",
            f"${costs['grok-3']:.2f}",
            f"${costs['grok-4']:.2f}",
        )

    console.print(projection_table)
    console.print("[dim]Based on 500 input + 200 output tokens per request[/dim]")

    # Summary
    console.print("\n[bold yellow]Cost Optimization Checklist:[/bold yellow]")
    console.print(
        """
  [ ] Use cheapest model that meets quality requirements
  [ ] Optimize prompts for conciseness
  [ ] Set appropriate max_tokens limits
  [ ] Implement response caching
  [ ] Batch similar requests when possible
  [ ] Monitor usage and set budget alerts
  [ ] Review and prune conversation history
  [ ] Use async/parallel for throughput
"""
    )


if __name__ == "__main__":
    main()
