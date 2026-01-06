#!/usr/bin/env python3
"""
02_retries.py - Retry Strategies with Exponential Backoff

This example demonstrates various retry strategies for handling
transient failures when using the xAI API, including exponential
backoff, jitter, and circuit breaker patterns.

Key concepts:
- Exponential backoff algorithm
- Adding jitter to prevent thundering herd
- Circuit breaker pattern
- Retry budgets and limits
"""

import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
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
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_mode: str = "full"  # "none", "full", "equal", "decorrelated"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1

    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    last_failure_time: float = 0
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failures = 0
        self.half_open_calls = 0

    def record_failure(self):
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.half_open_calls = 0

        elif self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate backoff delay with configurable jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds.
    """
    # Calculate base exponential delay
    exp_delay = config.base_delay * (config.exponential_base**attempt)
    exp_delay = min(exp_delay, config.max_delay)

    # Apply jitter based on mode
    if config.jitter_mode == "none":
        return exp_delay

    elif config.jitter_mode == "full":
        # Random between 0 and exp_delay
        return random.uniform(0, exp_delay)

    elif config.jitter_mode == "equal":
        # Half exponential, half random
        return exp_delay / 2 + random.uniform(0, exp_delay / 2)

    elif config.jitter_mode == "decorrelated":
        # Each delay depends on previous (simplified)
        return min(config.max_delay, random.uniform(config.base_delay, exp_delay * 3))

    return exp_delay


def retry_with_backoff(config: RetryConfig | None = None):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries):
                try:
                    return func(*args, **kwargs)

                except RateLimitError as e:
                    last_exception = e
                    if attempt < config.max_retries - 1:
                        delay = calculate_backoff(attempt, config)
                        console.print(
                            f"[yellow]Retry {attempt + 1}/{config.max_retries}, "
                            f"waiting {delay:.2f}s[/yellow]"
                        )
                        time.sleep(delay)

                except Exception:
                    # Non-retryable error
                    raise

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


def retry_with_circuit_breaker(
    func,
    circuit: CircuitBreaker,
    config: RetryConfig | None = None,
):
    """
    Execute a function with retry and circuit breaker protection.

    Args:
        func: Function to execute.
        circuit: Circuit breaker instance.
        config: Retry configuration.

    Returns:
        Function result or raises exception.
    """
    if config is None:
        config = RetryConfig()

    if not circuit.can_execute():
        raise Exception(f"Circuit breaker is {circuit.state.value}")

    if circuit.state == CircuitState.HALF_OPEN:
        circuit.half_open_calls += 1

    last_exception = None

    for attempt in range(config.max_retries):
        try:
            result = func()
            circuit.record_success()
            return result

        except RateLimitError as e:
            last_exception = e
            if attempt < config.max_retries - 1:
                delay = calculate_backoff(attempt, config)
                time.sleep(delay)

        except Exception:
            circuit.record_failure()
            raise

    circuit.record_failure()
    raise last_exception


def main():
    console.print(
        Panel.fit(
            "[bold blue]Retry Strategies Example[/bold blue]\n"
            "Exponential backoff and circuit breakers",
            border_style="blue",
        )
    )

    # Check for API key
    if not os.environ.get("X_AI_API_KEY"):
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Backoff strategies
    console.print("\n[bold yellow]Backoff Strategies:[/bold yellow]")

    backoff_table = Table(show_header=True, header_style="bold cyan")
    backoff_table.add_column("Strategy", style="green")
    backoff_table.add_column("Formula")
    backoff_table.add_column("When to Use")

    backoff_table.add_row(
        "No jitter",
        "delay = base * 2^attempt",
        "Simple cases, single client",
    )
    backoff_table.add_row(
        "Full jitter",
        "delay = random(0, base * 2^attempt)",
        "Distributed systems",
    )
    backoff_table.add_row(
        "Equal jitter",
        "delay = (base * 2^attempt)/2 + random",
        "Balance of predictability",
    )
    backoff_table.add_row(
        "Decorrelated",
        "delay = random(base, prev_delay * 3)",
        "AWS recommendation",
    )

    console.print(backoff_table)

    # Demonstrate backoff calculations
    console.print("\n[bold yellow]Backoff Delay Examples:[/bold yellow]")

    delay_table = Table(show_header=True, header_style="bold cyan")
    delay_table.add_column("Attempt", style="green")
    delay_table.add_column("No Jitter")
    delay_table.add_column("Full Jitter")
    delay_table.add_column("Equal Jitter")

    for attempt in range(6):
        no_jitter = calculate_backoff(attempt, RetryConfig(jitter_mode="none"))
        full_jitter = calculate_backoff(attempt, RetryConfig(jitter_mode="full"))
        equal_jitter = calculate_backoff(attempt, RetryConfig(jitter_mode="equal"))

        delay_table.add_row(
            str(attempt + 1),
            f"{no_jitter:.2f}s",
            f"{full_jitter:.2f}s",
            f"{equal_jitter:.2f}s",
        )

    console.print(delay_table)

    # Example 1: Simple retry with decorator
    console.print("\n[bold yellow]Example 1: Retry with Decorator[/bold yellow]")

    @retry_with_backoff(RetryConfig(max_retries=3, jitter_mode="full"))
    def make_request():
        return client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": "Say 'hello'"}],
            max_tokens=10,
        )

    try:
        response = make_request()
        console.print(f"[green]Success:[/green] {response.choices[0].message.content}")
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")

    # Example 2: Circuit breaker
    console.print("\n[bold yellow]Example 2: Circuit Breaker Pattern[/bold yellow]")

    circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

    console.print(f"Circuit state: [cyan]{circuit.state.value}[/cyan]")

    def api_call():
        return client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": "Hi!"}],
            max_tokens=10,
        )

    try:
        result = retry_with_circuit_breaker(api_call, circuit)
        console.print(f"[green]Success:[/green] {result.choices[0].message.content}")
        console.print(f"Circuit state: [cyan]{circuit.state.value}[/cyan]")
    except Exception as e:
        console.print(f"[yellow]Error:[/yellow] {e}")
        console.print(f"Circuit state: [cyan]{circuit.state.value}[/cyan]")

    # Circuit breaker states
    console.print("\n[bold yellow]Circuit Breaker States:[/bold yellow]")

    states_table = Table(show_header=True, header_style="bold cyan")
    states_table.add_column("State", style="green")
    states_table.add_column("Description")
    states_table.add_column("Action")

    states_table.add_row(
        "CLOSED",
        "Normal operation",
        "Allow all requests",
    )
    states_table.add_row(
        "OPEN",
        "Too many failures",
        "Reject requests immediately",
    )
    states_table.add_row(
        "HALF_OPEN",
        "Testing recovery",
        "Allow limited requests",
    )

    console.print(states_table)

    # Best practices
    console.print("\n[bold yellow]Retry Best Practices:[/bold yellow]")
    console.print(
        """
  [cyan]1. Use exponential backoff:[/cyan]
     Prevents overwhelming a recovering service

  [cyan]2. Add jitter:[/cyan]
     Prevents synchronized retries from multiple clients

  [cyan]3. Set maximum retries:[/cyan]
     Don't retry forever (3-5 attempts typical)

  [cyan]4. Cap maximum delay:[/cyan]
     Prevent unreasonably long waits (30-60s typical)

  [cyan]5. Use circuit breakers:[/cyan]
     Fail fast when service is unhealthy

  [cyan]6. Only retry transient errors:[/cyan]
     Don't retry auth or validation errors

  [cyan]7. Log retry attempts:[/cyan]
     Monitor retry patterns for issues
"""
    )

    # Code template
    console.print("\n[bold yellow]Retry Template:[/bold yellow]")
    console.print(
        """
[dim]import time
import random

def exponential_backoff_retry(func, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt < max_retries - 1:
                # Exponential backoff with full jitter
                exp_delay = base_delay * (2 ** attempt)
                delay = random.uniform(0, min(exp_delay, 60))
                time.sleep(delay)
    raise Exception("Max retries exceeded")

# Usage
result = exponential_backoff_retry(
    lambda: client.chat.completions.create(...)
)[/dim]
"""
    )


if __name__ == "__main__":
    main()
