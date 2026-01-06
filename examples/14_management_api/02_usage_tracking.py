#!/usr/bin/env python3
"""
02_usage_tracking.py - Track API Usage

This example demonstrates how to track and analyze API usage,
including token consumption, costs, and request patterns.

Key concepts:
- Tracking usage from responses
- Calculating costs
- Usage analytics
- Budget monitoring
"""

import os
from dataclasses import dataclass, field
from datetime import datetime

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


@dataclass
class UsageRecord:
    """Record of a single API request's usage."""

    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    cost: float = 0.0


@dataclass
class UsageTracker:
    """Track and analyze API usage over time."""

    records: list[UsageRecord] = field(default_factory=list)
    budget_limit: float | None = None

    # Pricing per million tokens (approximate - check docs.x.ai)
    PRICING = {
        "grok-4-1-fast-reasoning": {"input": 0.30, "output": 0.50},
        "grok-3": {"input": 3.00, "output": 15.00},
        "grok-4": {"input": 2.00, "output": 10.00},
        "grok-4-fast": {"input": 5.00, "output": 25.00},
    }

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate the cost for a request."""
        pricing = self.PRICING.get(model, self.PRICING["grok-4-1-fast-reasoning"])

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def record_usage(self, response, model: str):
        """Record usage from an API response."""
        usage = response.usage

        cost = self.calculate_cost(
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
        )

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=(
                getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0)
                if hasattr(usage, "prompt_tokens_details")
                else 0
            ),
            cost=cost,
        )

        self.records.append(record)

        # Check budget
        if self.budget_limit and self.get_total_cost() > self.budget_limit:
            console.print(f"[red]Warning: Budget limit (${self.budget_limit:.2f}) exceeded![/red]")

        return record

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(r.total_tokens for r in self.records)

    def get_total_cost(self) -> float:
        """Get total cost."""
        return sum(r.cost for r in self.records)

    def get_usage_by_model(self) -> dict:
        """Get usage breakdown by model."""
        usage = {}
        for record in self.records:
            if record.model not in usage:
                usage[record.model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                }
            usage[record.model]["requests"] += 1
            usage[record.model]["tokens"] += record.total_tokens
            usage[record.model]["cost"] += record.cost
        return usage

    def get_summary(self) -> dict:
        """Get usage summary."""
        return {
            "total_requests": len(self.records),
            "total_tokens": self.get_total_tokens(),
            "total_cost": self.get_total_cost(),
            "avg_tokens_per_request": (
                self.get_total_tokens() / len(self.records) if self.records else 0
            ),
            "by_model": self.get_usage_by_model(),
        }


# Global tracker instance
tracker = UsageTracker()


def tracked_chat(prompt: str, model: str = "grok-4-1-fast-reasoning", **kwargs) -> str:
    """
    Make a chat request with automatic usage tracking.

    Args:
        prompt: The user prompt.
        model: Model to use.
        **kwargs: Additional parameters.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )

    # Record usage
    tracker.record_usage(response, model)

    return response.choices[0].message.content


def display_usage_summary():
    """Display a formatted usage summary."""
    summary = tracker.get_summary()

    console.print("\n[bold cyan]Usage Summary:[/bold cyan]")

    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="green")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Requests", str(summary["total_requests"]))
    summary_table.add_row("Total Tokens", f"{summary['total_tokens']:,}")
    summary_table.add_row("Total Cost", f"${summary['total_cost']:.6f}")
    summary_table.add_row("Avg Tokens/Request", f"{summary['avg_tokens_per_request']:.1f}")

    console.print(summary_table)

    # By model
    if summary["by_model"]:
        console.print("\n[bold cyan]Usage by Model:[/bold cyan]")

        model_table = Table(show_header=True, header_style="bold cyan")
        model_table.add_column("Model", style="green")
        model_table.add_column("Requests", justify="right")
        model_table.add_column("Tokens", justify="right")
        model_table.add_column("Cost", justify="right")

        for model, data in summary["by_model"].items():
            model_table.add_row(
                model,
                str(data["requests"]),
                f"{data['tokens']:,}",
                f"${data['cost']:.6f}",
            )

        console.print(model_table)


def main():
    console.print(
        Panel.fit(
            "[bold blue]Usage Tracking Example[/bold blue]\nMonitor API consumption and costs",
            border_style="blue",
        )
    )

    # Check for API key
    if not os.environ.get("X_AI_API_KEY"):
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Usage tracking overview
    console.print("\n[bold yellow]Why Track Usage?[/bold yellow]")
    console.print(
        """
  Tracking API usage helps you:

  - [cyan]Control costs:[/cyan] Monitor spending against budget
  - [cyan]Optimize efficiency:[/cyan] Identify expensive patterns
  - [cyan]Debug issues:[/cyan] Understand token consumption
  - [cyan]Plan capacity:[/cyan] Forecast future needs
"""
    )

    # Set a budget limit
    tracker.budget_limit = 0.10  # $0.10 for demo
    console.print(f"[dim]Budget limit set to ${tracker.budget_limit:.2f}[/dim]")

    # Example 1: Make tracked requests
    console.print("\n[bold yellow]Example 1: Tracked Requests[/bold yellow]")

    prompts = [
        ("Hello!", "grok-4-1-fast-reasoning"),
        ("What is 2+2?", "grok-4-1-fast-reasoning"),
        ("Explain recursion briefly.", "grok-4-1-fast-reasoning"),
    ]

    for prompt, model in prompts:
        console.print(f"\n[bold green]Prompt:[/bold green] {prompt}")
        response = tracked_chat(prompt, model, max_tokens=50)
        console.print(f"[bold cyan]Response:[/bold cyan] {response[:100]}...")

        # Show last record
        record = tracker.records[-1]
        console.print(f"[dim]Tokens: {record.total_tokens} | Cost: ${record.cost:.6f}[/dim]")

    # Display summary
    display_usage_summary()

    # Example 2: Usage analytics
    console.print("\n[bold yellow]Example 2: Recent Usage Details[/bold yellow]")

    records_table = Table(show_header=True, header_style="bold cyan")
    records_table.add_column("#", style="green")
    records_table.add_column("Time")
    records_table.add_column("Model")
    records_table.add_column("Prompt Tokens", justify="right")
    records_table.add_column("Completion Tokens", justify="right")
    records_table.add_column("Cost", justify="right")

    for i, record in enumerate(tracker.records[-5:], 1):
        records_table.add_row(
            str(i),
            record.timestamp.strftime("%H:%M:%S"),
            record.model,
            str(record.prompt_tokens),
            str(record.completion_tokens),
            f"${record.cost:.6f}",
        )

    console.print(records_table)

    # Cost estimation
    console.print("\n[bold yellow]Cost Estimation:[/bold yellow]")

    estimate_table = Table(show_header=True, header_style="bold cyan")
    estimate_table.add_column("Scenario", style="green")
    estimate_table.add_column("Requests/Day")
    estimate_table.add_column("Est. Daily Cost")
    estimate_table.add_column("Est. Monthly Cost")

    scenarios = [
        ("Light usage", 100, "grok-4-1-fast-reasoning"),
        ("Medium usage", 1000, "grok-4-1-fast-reasoning"),
        ("Heavy usage", 10000, "grok-4-1-fast-reasoning"),
        ("Premium usage", 1000, "grok-4"),
    ]

    avg_tokens = 500  # Average tokens per request

    for name, requests, model in scenarios:
        pricing = tracker.PRICING.get(model, tracker.PRICING["grok-4-1-fast-reasoning"])
        daily_tokens = requests * avg_tokens
        daily_cost = (daily_tokens / 1_000_000) * (pricing["input"] + pricing["output"])
        monthly_cost = daily_cost * 30

        estimate_table.add_row(
            f"{name} ({model})",
            str(requests),
            f"${daily_cost:.2f}",
            f"${monthly_cost:.2f}",
        )

    console.print(estimate_table)

    # Usage response fields
    console.print("\n[bold yellow]Usage Response Fields:[/bold yellow]")
    console.print(
        """
  Every API response includes a [cyan]usage[/cyan] object:

  [dim]{
    "prompt_tokens": 15,           // Input tokens
    "completion_tokens": 50,       // Output tokens
    "total_tokens": 65,            // Total tokens
    "prompt_tokens_details": {
      "cached_tokens": 0           // Tokens from cache (cheaper)
    }
  }[/dim]

  Use this data to track consumption per request.
"""
    )

    # Best practices
    console.print("\n[bold yellow]Best Practices:[/bold yellow]")
    console.print(
        """
  [cyan]1. Track every request:[/cyan]
     Wrap API calls to automatically record usage.

  [cyan]2. Set budget alerts:[/cyan]
     Monitor costs and alert when approaching limits.

  [cyan]3. Analyze patterns:[/cyan]
     Identify expensive operations and optimize.

  [cyan]4. Use caching:[/cyan]
     Repeated prompts may benefit from cached tokens.

  [cyan]5. Right-size models:[/cyan]
     Use cheaper models for simple tasks.

  [cyan]6. Export data:[/cyan]
     Save usage data for long-term analysis.
"""
    )

    # Code template
    console.print("\n[bold yellow]Code Template:[/bold yellow]")
    console.print(
        """
[dim]class UsageTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0

    def track(self, response, model):
        usage = response.usage
        cost = self.calculate_cost(model, usage)
        self.total_tokens += usage.total_tokens
        self.total_cost += cost
        return cost

    def calculate_cost(self, model, usage):
        # Model-specific pricing
        prices = {"grok-4-1-fast-reasoning": (0.30, 0.50)}
        input_price, output_price = prices.get(model, (0.30, 0.50))
        return (
            (usage.prompt_tokens / 1e6) * input_price +
            (usage.completion_tokens / 1e6) * output_price
        )

# Usage
tracker = UsageTracker()
response = client.chat.completions.create(...)
tracker.track(response, "grok-4-1-fast-reasoning")[/dim]
"""
    )


if __name__ == "__main__":
    main()
