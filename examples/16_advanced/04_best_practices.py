#!/usr/bin/env python3
"""
04_best_practices.py - Best Practices Summary

This example provides a comprehensive summary of best practices for
using the xAI API effectively, covering security, performance,
reliability, and code quality.

Key concepts:
- Security best practices
- Performance optimization
- Reliability patterns
- Code organization
- Monitoring and observability
"""

from dataclasses import dataclass

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


@dataclass
class BestPractice:
    """A best practice recommendation."""

    category: str
    title: str
    description: str
    example: str
    priority: str  # "critical", "high", "medium"


BEST_PRACTICES = [
    # Security
    BestPractice(
        category="Security",
        title="Secure API Key Storage",
        description="Never hardcode API keys. Use environment variables or secrets managers.",
        example='api_key = os.environ["X_AI_API_KEY"]',
        priority="critical",
    ),
    BestPractice(
        category="Security",
        title="Validate User Input",
        description="Sanitize and validate all user input before sending to API.",
        example="prompt = sanitize_input(user_input)",
        priority="critical",
    ),
    BestPractice(
        category="Security",
        title="Implement Rate Limiting",
        description="Protect your application from abuse with rate limiting.",
        example="@rate_limit(calls=100, period=60)",
        priority="high",
    ),
    # Performance
    BestPractice(
        category="Performance",
        title="Use Streaming for Long Responses",
        description="Stream responses for better user experience and memory efficiency.",
        example="stream=True in chat.completions.create()",
        priority="high",
    ),
    BestPractice(
        category="Performance",
        title="Set Appropriate Timeouts",
        description="Configure timeouts to prevent hanging requests.",
        example="timeout=30.0 in API calls",
        priority="high",
    ),
    BestPractice(
        category="Performance",
        title="Use Async for Concurrency",
        description="Use async/await for concurrent API calls.",
        example="async with client.chat.completions.create(...)",
        priority="medium",
    ),
    # Reliability
    BestPractice(
        category="Reliability",
        title="Implement Retry Logic",
        description="Use exponential backoff for transient failures.",
        example="@retry(max_attempts=3, backoff=exponential)",
        priority="critical",
    ),
    BestPractice(
        category="Reliability",
        title="Handle All Error Types",
        description="Catch and handle specific exceptions appropriately.",
        example="except RateLimitError: sleep(backoff)",
        priority="critical",
    ),
    BestPractice(
        category="Reliability",
        title="Use Circuit Breakers",
        description="Prevent cascade failures with circuit breakers.",
        example="if circuit.is_open: return fallback",
        priority="medium",
    ),
    # Cost
    BestPractice(
        category="Cost",
        title="Choose Appropriate Models",
        description="Use cheaper models for simple tasks.",
        example="grok-4-1-fast-reasoning for classification, grok-4 for complex reasoning",
        priority="high",
    ),
    BestPractice(
        category="Cost",
        title="Optimize Token Usage",
        description="Write concise prompts and set max_tokens.",
        example="max_tokens=500 to limit output",
        priority="high",
    ),
    BestPractice(
        category="Cost",
        title="Cache Responses",
        description="Cache identical requests to reduce API calls.",
        example="@lru_cache for repeated prompts",
        priority="medium",
    ),
    # Code Quality
    BestPractice(
        category="Code Quality",
        title="Use Type Hints",
        description="Add type annotations for better code quality.",
        example="def chat(prompt: str) -> str:",
        priority="medium",
    ),
    BestPractice(
        category="Code Quality",
        title="Centralize Configuration",
        description="Keep API configuration in one place.",
        example="from config import api_client",
        priority="medium",
    ),
    BestPractice(
        category="Code Quality",
        title="Log Appropriately",
        description="Log errors and important events for debugging.",
        example="logger.info(f'Tokens used: {usage.total_tokens}')",
        priority="high",
    ),
]


def main():
    console.print(
        Panel.fit(
            "[bold blue]Best Practices Summary[/bold blue]\nComprehensive guide for xAI API usage",
            border_style="blue",
        )
    )

    # Overview
    console.print("\n[bold yellow]Overview:[/bold yellow]")
    console.print(
        """
  This guide covers best practices across five key areas:

  1. [cyan]Security[/cyan] - Protecting your application and data
  2. [cyan]Performance[/cyan] - Optimizing speed and efficiency
  3. [cyan]Reliability[/cyan] - Building robust, fault-tolerant systems
  4. [cyan]Cost[/cyan] - Minimizing expenses while maintaining quality
  5. [cyan]Code Quality[/cyan] - Writing maintainable, professional code
"""
    )

    # Group practices by category
    categories = {}
    for practice in BEST_PRACTICES:
        if practice.category not in categories:
            categories[practice.category] = []
        categories[practice.category].append(practice)

    # Display by category
    for category, practices in categories.items():
        console.print(f"\n[bold yellow]{category} Best Practices:[/bold yellow]")

        for p in practices:
            priority_color = {
                "critical": "red",
                "high": "yellow",
                "medium": "cyan",
            }.get(p.priority, "white")

            console.print(
                f"\n  [{priority_color}][{p.priority.upper()}][/{priority_color}] "
                f"[bold]{p.title}[/bold]"
            )
            console.print(f"  {p.description}")
            console.print(f"  [dim]Example: {p.example}[/dim]")

    # Quick reference table
    console.print("\n[bold yellow]Quick Reference:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Do", style="green")
    ref_table.add_column("Don't", style="red")

    ref_table.add_row(
        "Store API keys in environment variables",
        "Hardcode API keys in source code",
    )
    ref_table.add_row(
        "Use exponential backoff for retries",
        "Retry immediately or indefinitely",
    )
    ref_table.add_row(
        "Set timeouts on all API calls",
        "Let requests hang indefinitely",
    )
    ref_table.add_row(
        "Match model to task complexity",
        "Use expensive models for simple tasks",
    )
    ref_table.add_row(
        "Validate and sanitize user input",
        "Pass raw user input to API",
    )
    ref_table.add_row(
        "Log errors with context",
        "Silently catch and ignore errors",
    )
    ref_table.add_row(
        "Use streaming for long responses",
        "Wait for full response for large outputs",
    )
    ref_table.add_row(
        "Cache identical requests",
        "Make duplicate API calls",
    )

    console.print(ref_table)

    # Production checklist
    console.print("\n[bold yellow]Production Readiness Checklist:[/bold yellow]")
    console.print(
        """
  [bold]Security:[/bold]
  [ ] API keys stored securely (env vars/secrets manager)
  [ ] Input validation implemented
  [ ] Rate limiting configured
  [ ] No sensitive data in logs

  [bold]Reliability:[/bold]
  [ ] Retry logic with exponential backoff
  [ ] Specific exception handling
  [ ] Timeouts configured
  [ ] Circuit breakers for critical paths
  [ ] Graceful degradation/fallbacks

  [bold]Performance:[/bold]
  [ ] Streaming for long responses
  [ ] Async for concurrent requests
  [ ] Response caching implemented
  [ ] Connection pooling (if applicable)

  [bold]Monitoring:[/bold]
  [ ] Error logging configured
  [ ] Usage metrics tracked
  [ ] Cost monitoring set up
  [ ] Alerts for anomalies

  [bold]Testing:[/bold]
  [ ] Unit tests with mocked API
  [ ] Integration tests
  [ ] Load testing performed
  [ ] Error scenarios tested
"""
    )

    # Code template
    console.print("\n[bold yellow]Production-Ready Template:[/bold yellow]")
    console.print(
        """
[dim]import os
import logging
from functools import lru_cache
from openai import OpenAI, RateLimitError
import time

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
    timeout=30.0,
)

# Cached chat function
@lru_cache(maxsize=100)
def _cached_chat(prompt: str, model: str) -> str:
    return _chat_with_retry(prompt, model)

# Retry wrapper
def _chat_with_retry(prompt: str, model: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            logger.info(f"Tokens: {response.usage.total_tokens}")
            return response.choices[0].message.content
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return ""

# Public interface
def chat(prompt: str, model: str = "grok-4-1-fast-reasoning", cache: bool = True) -> str:
    '''
    Chat with Grok using best practices.

    Args:
        prompt: User prompt (will be validated)
        model: Model to use
        cache: Whether to cache response

    Returns:
        Assistant response
    '''
    # Validate input
    if not prompt or len(prompt) > 10000:
        raise ValueError("Invalid prompt")

    if cache:
        return _cached_chat(prompt, model)
    return _chat_with_retry(prompt, model)[/dim]
"""
    )

    # Resources
    console.print("\n[bold yellow]Resources:[/bold yellow]")
    console.print(
        """
  [cyan]Documentation:[/cyan]
  - xAI Docs: https://docs.x.ai
  - API Reference: https://docs.x.ai/docs/api-reference
  - xAI Console: https://console.x.ai

  [cyan]SDKs:[/cyan]
  - OpenAI Python SDK: pip install openai
  - OpenAI JS SDK: npm install openai

  [cyan]Community:[/cyan]
  - xAI Discord/Community forums
  - GitHub issues for SDK bugs

  [cyan]Support:[/cyan]
  - support@x.ai for account issues
  - Rate limit increases by request
"""
    )

    # Summary
    console.print("\n[bold yellow]Summary:[/bold yellow]")
    console.print(
        """
  Following these best practices will help you build:

  - [green]Secure[/green] applications that protect data
  - [green]Reliable[/green] systems that handle failures gracefully
  - [green]Performant[/green] services that scale efficiently
  - [green]Cost-effective[/green] solutions that stay within budget
  - [green]Maintainable[/green] code that's easy to update

  Start with the critical items and progressively implement
  the rest as your application matures.
"""
    )


if __name__ == "__main__":
    main()
