#!/usr/bin/env python3
"""
03_rate_limiting.py - Handle Rate Limits Gracefully

This example demonstrates strategies for handling rate limits when
using the xAI API, including retry logic, backoff strategies, and
request management.

Key concepts:
- Understanding rate limit errors (429)
- Exponential backoff with jitter
- Request queuing
- Rate limit headers
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


@dataclass
class RateLimitConfig:
    """Configuration for rate limit handling."""

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


def exponential_backoff_with_jitter(
    attempt: int,
    config: RateLimitConfig,
) -> float:
    """
    Calculate delay with exponential backoff and optional jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        config: Rate limit configuration.

    Returns:
        Delay in seconds.
    """
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter between 0-50% of delay
        jitter = random.uniform(0, delay * 0.5)
        delay += jitter

    return delay


def make_request_with_retry(
    prompt: str,
    config: RateLimitConfig | None = None,
) -> dict:
    """
    Make an API request with automatic retry on rate limit.

    Args:
        prompt: The prompt to send.
        config: Optional rate limit configuration.

    Returns:
        Dictionary with response or error info.
    """
    if config is None:
        config = RateLimitConfig()

    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "attempts": attempt + 1,
            }

        except RateLimitError:
            if attempt < config.max_retries - 1:
                delay = exponential_backoff_with_jitter(attempt, config)
                console.print(
                    f"[yellow]Rate limited (attempt {attempt + 1}). "
                    f"Retrying in {delay:.2f}s...[/yellow]"
                )
                time.sleep(delay)
            else:
                return {
                    "success": False,
                    "error": "Rate limit exceeded after max retries",
                    "attempts": attempt + 1,
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "attempts": attempt + 1,
            }

    return {"success": False, "error": "Unknown error", "attempts": config.max_retries}


class RequestQueue:
    """
    A simple request queue with rate limiting.

    This helps manage multiple requests while respecting rate limits.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times: deque = deque()
        self.token_usage: deque = deque()

    def can_make_request(self, estimated_tokens: int = 1000) -> tuple[bool, float]:
        """
        Check if a request can be made within rate limits.

        Args:
            estimated_tokens: Estimated tokens for the request.

        Returns:
            Tuple of (can_proceed, wait_time).
        """
        now = time.time()
        minute_ago = now - 60

        # Clean old entries
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
        while self.token_usage and self.token_usage[0][0] < minute_ago:
            self.token_usage.popleft()

        # Check request limit
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = self.request_times[0] - minute_ago
            return False, wait_time

        # Check token limit
        current_tokens = sum(t[1] for t in self.token_usage)
        if current_tokens + estimated_tokens > self.tokens_per_minute:
            wait_time = self.token_usage[0][0] - minute_ago
            return False, wait_time

        return True, 0

    def record_request(self, tokens_used: int):
        """Record a completed request."""
        now = time.time()
        self.request_times.append(now)
        self.token_usage.append((now, tokens_used))

    def process_request(self, prompt: str) -> dict:
        """
        Process a request with rate limiting.

        Args:
            prompt: The prompt to send.

        Returns:
            Response dictionary.
        """
        estimated_tokens = len(prompt.split()) * 2  # Rough estimate

        can_proceed, wait_time = self.can_make_request(estimated_tokens)

        if not can_proceed:
            console.print(f"[yellow]Rate limit approaching. Waiting {wait_time:.1f}s...[/yellow]")
            time.sleep(wait_time)

        result = make_request_with_retry(prompt)

        if result["success"]:
            # Record actual usage (estimate if not available)
            self.record_request(estimated_tokens)

        return result


def batch_with_rate_limit(
    prompts: list[str],
    delay_between: float = 0.5,
) -> list[dict]:
    """
    Process multiple prompts with rate limiting.

    Args:
        prompts: List of prompts to process.
        delay_between: Delay between requests in seconds.

    Returns:
        List of results.
    """
    results = []
    queue = RequestQueue()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing requests...", total=len(prompts))

        for i, prompt in enumerate(prompts):
            result = queue.process_request(prompt)
            results.append(result)

            progress.update(task, advance=1, description=f"Request {i + 1}/{len(prompts)}")

            if i < len(prompts) - 1:
                time.sleep(delay_between)

    return results


def main():
    console.print(
        Panel.fit(
            "[bold blue]Rate Limiting Example[/bold blue]\nHandle API rate limits gracefully",
            border_style="blue",
        )
    )

    # Check for API key
    if not os.environ.get("X_AI_API_KEY"):
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Rate limit overview
    console.print("\n[bold yellow]Understanding Rate Limits:[/bold yellow]")
    console.print(
        """
  Rate limits protect the API from overload and ensure fair usage.
  xAI enforces limits on:

  - [cyan]Requests per minute (RPM):[/cyan] Number of API calls
  - [cyan]Tokens per minute (TPM):[/cyan] Total tokens processed

  When exceeded, you receive a [red]429 Too Many Requests[/red] error.
"""
    )

    # Rate limit tiers (approximate - check docs.x.ai)
    console.print("\n[bold yellow]Rate Limit Tiers:[/bold yellow]")

    tiers_table = Table(show_header=True, header_style="bold cyan")
    tiers_table.add_column("Tier", style="green")
    tiers_table.add_column("Requests/Min")
    tiers_table.add_column("Tokens/Min")
    tiers_table.add_column("Notes")

    tiers_table.add_row("Free", "60", "40,000", "Default tier")
    tiers_table.add_row("Basic", "500", "200,000", "Paid tier")
    tiers_table.add_row("Pro", "2,000", "1,000,000", "Higher usage")
    tiers_table.add_row("Enterprise", "Custom", "Custom", "Contact xAI")

    console.print(tiers_table)
    console.print("[dim]Note: Check docs.x.ai for current limits[/dim]")

    # Example 1: Simple retry
    console.print("\n[bold yellow]Example 1: Request with Automatic Retry[/bold yellow]")

    result = make_request_with_retry("Say hello in one word.")

    if result["success"]:
        console.print(f"[green]Success![/green] Response: {result['content']}")
        console.print(f"[dim]Completed in {result['attempts']} attempt(s)[/dim]")
    else:
        console.print(f"[red]Failed:[/red] {result['error']}")

    # Example 2: Batch processing
    console.print("\n[bold yellow]Example 2: Batch Processing with Rate Limiting[/bold yellow]")

    prompts = [
        "What is 2+2?",
        "Name a color.",
        "Say yes or no.",
    ]

    console.print(f"[dim]Processing {len(prompts)} requests...[/dim]")

    results = batch_with_rate_limit(prompts, delay_between=0.5)

    for i, result in enumerate(results):
        status = "[green]OK[/green]" if result["success"] else "[red]FAIL[/red]"
        content = result.get("content", result.get("error", "Unknown"))[:50]
        console.print(f"  {i + 1}. {status} - {content}")

    # Backoff strategies
    console.print("\n[bold yellow]Backoff Strategies:[/bold yellow]")

    strategies_table = Table(show_header=True, header_style="bold cyan")
    strategies_table.add_column("Strategy", style="green")
    strategies_table.add_column("Formula")
    strategies_table.add_column("Pros/Cons")

    strategies_table.add_row(
        "Fixed Delay",
        "delay = constant",
        "Simple but inefficient",
    )
    strategies_table.add_row(
        "Linear Backoff",
        "delay = base * attempt",
        "Gradual increase",
    )
    strategies_table.add_row(
        "Exponential",
        "delay = base * 2^attempt",
        "Quick recovery, may overshoot",
    )
    strategies_table.add_row(
        "Exp + Jitter",
        "delay = (base * 2^attempt) + random",
        "Best for distributed systems",
    )

    console.print(strategies_table)

    # Show backoff example
    console.print("\n[bold yellow]Exponential Backoff Example:[/bold yellow]")

    config = RateLimitConfig(base_delay=1.0, max_delay=60.0)

    backoff_table = Table(show_header=True, header_style="bold cyan")
    backoff_table.add_column("Attempt", style="green")
    backoff_table.add_column("Base Delay", justify="right")
    backoff_table.add_column("With Jitter (example)", justify="right")

    for i in range(6):
        base = config.base_delay * (config.exponential_base**i)
        base = min(base, config.max_delay)
        with_jitter = exponential_backoff_with_jitter(i, config)
        backoff_table.add_row(str(i + 1), f"{base:.1f}s", f"{with_jitter:.1f}s")

    console.print(backoff_table)

    # Best practices
    console.print("\n[bold yellow]Best Practices:[/bold yellow]")
    console.print(
        """
  [cyan]1. Use exponential backoff with jitter:[/cyan]
     Prevents thundering herd problem in distributed systems.

  [cyan]2. Set reasonable max retries:[/cyan]
     3-5 retries is usually sufficient. Don't retry forever.

  [cyan]3. Track rate limit headers:[/cyan]
     Some APIs return remaining quota in headers. Use it!

  [cyan]4. Implement request queuing:[/cyan]
     For high-volume apps, queue requests and process steadily.

  [cyan]5. Contact support for higher limits:[/cyan]
     If you consistently hit limits, request a higher tier.

  [cyan]6. Batch when possible:[/cyan]
     Combine multiple small requests into larger batches.
"""
    )

    # Code template
    console.print("\n[bold yellow]Code Template:[/bold yellow]")
    console.print(
        """
[dim]import time
import random
from openai import RateLimitError

def request_with_backoff(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[{"role": "user", "content": prompt}]
            )
        except RateLimitError:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise
    return None[/dim]
"""
    )


if __name__ == "__main__":
    main()
