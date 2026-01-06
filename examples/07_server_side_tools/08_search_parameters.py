#!/usr/bin/env python3
"""
08_search_parameters.py - Advanced Search Parameters and Filtering

This example demonstrates advanced configuration options for xAI's
server-side search tools. Fine-tuning search parameters helps you
get more relevant and targeted results.

Key concepts:
- Search mode configurations
- Source-specific parameters
- Query optimization techniques
- Combining search with other request parameters
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

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


def search_with_system_context(client: OpenAI, query: str, context: str) -> str:
    """
    Perform search with system-level context to guide results.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.
        context: System context to guide the search interpretation.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query},
        ],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}, {"type": "news"}],
            }
        },
    )

    return response.choices[0].message.content


def search_with_temperature(client: OpenAI, query: str, temperature: float) -> str:
    """
    Perform search with specific temperature setting.

    Lower temperature = more focused, deterministic responses
    Higher temperature = more creative, varied responses

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.
        temperature: Temperature value (0.0 to 2.0).

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}],
            }
        },
    )

    return response.choices[0].message.content


def search_with_max_tokens(client: OpenAI, query: str, max_tokens: int) -> str:
    """
    Perform search with token limit for response length control.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        max_tokens=max_tokens,
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}],
            }
        },
    )

    return response.choices[0].message.content


def targeted_search_by_domain(client: OpenAI, query: str, domain_hint: str) -> str:
    """
    Perform search with domain-specific guidance.

    By including domain context in the query, you can guide the search
    toward more relevant sources.

    Args:
        client: The OpenAI client configured for xAI.
        query: The base search query.
        domain_hint: Domain or topic to focus on.

    Returns:
        The assistant's response.
    """
    enhanced_query = f"{query} (Focus on {domain_hint} sources and information)"

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": enhanced_query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}],
            }
        },
    )

    return response.choices[0].message.content


def compare_search_modes(client: OpenAI, query: str) -> dict:
    """
    Compare results from different search mode configurations.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        Dictionary with results from different modes.
    """
    results = {}

    # Auto mode
    response_auto = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "auto",
                "sources": [{"type": "web"}],
            }
        },
    )
    results["auto"] = response_auto.choices[0].message.content

    # Forced on mode
    response_on = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}],
            }
        },
    )
    results["on"] = response_on.choices[0].message.content

    return results


def main():
    console.print(
        Panel.fit(
            "[bold blue]Advanced Search Parameters and Filtering[/bold blue]\n"
            "Fine-tuning server-side search for better results",
            border_style="blue",
        )
    )

    client = get_client()

    # Example 1: Search with system context
    console.print("\n[bold cyan]Example 1: Search with System Context[/bold cyan]")
    console.print("[dim]Adding domain expertise context to guide search interpretation[/dim]")

    context = (
        "You are a financial analyst. When answering questions about markets "
        "and economics, prioritize data from reputable financial sources and "
        "provide analysis with specific numbers when available."
    )
    query1 = "What are the current trends in the stock market?"

    response1 = search_with_system_context(client, query1, context)
    console.print(
        Panel(Markdown(response1), title="Financial Context Search", border_style="green")
    )

    # Example 2: Temperature comparison
    console.print("\n[bold cyan]Example 2: Temperature Impact on Search Results[/bold cyan]")

    query2 = "What are some interesting applications of machine learning?"

    console.print("[dim]Low temperature (0.3) - More focused:[/dim]")
    response_low = search_with_temperature(client, query2, 0.3)
    console.print(
        Panel(
            Markdown(response_low[:400] + "..."),
            title="Temperature 0.3",
            border_style="blue",
        )
    )

    console.print("\n[dim]High temperature (1.0) - More varied:[/dim]")
    response_high = search_with_temperature(client, query2, 1.0)
    console.print(
        Panel(
            Markdown(response_high[:400] + "..."),
            title="Temperature 1.0",
            border_style="yellow",
        )
    )

    # Example 3: Response length control
    console.print("\n[bold cyan]Example 3: Response Length Control[/bold cyan]")

    query3 = "Explain quantum computing"

    console.print("[dim]Short response (100 tokens):[/dim]")
    response_short = search_with_max_tokens(client, query3, 100)
    console.print(Panel(response_short, title="100 Tokens", border_style="cyan"))

    console.print("\n[dim]Longer response (300 tokens):[/dim]")
    response_long = search_with_max_tokens(client, query3, 300)
    console.print(Panel(response_long, title="300 Tokens", border_style="green"))

    # Example 4: Domain-targeted search
    console.print("\n[bold cyan]Example 4: Domain-Targeted Search[/bold cyan]")

    base_query = "Latest developments in batteries"

    console.print("[dim]Scientific focus:[/dim]")
    response_science = targeted_search_by_domain(
        client, base_query, "scientific research and academic papers"
    )
    console.print(
        Panel(
            Markdown(response_science[:400] + "..."),
            title="Scientific Focus",
            border_style="green",
        )
    )

    console.print("\n[dim]Industry focus:[/dim]")
    response_industry = targeted_search_by_domain(
        client, base_query, "industry news and business applications"
    )
    console.print(
        Panel(
            Markdown(response_industry[:400] + "..."),
            title="Industry Focus",
            border_style="blue",
        )
    )

    # Show parameter reference table
    console.print("\n[bold yellow]Search Parameter Reference:[/bold yellow]")

    param_table = Table(title="Configuration Options")
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Type", style="yellow")
    param_table.add_column("Description", style="white")

    param_table.add_row(
        "mode",
        "string",
        '"auto", "on", or "off" - controls search behavior',
    )
    param_table.add_row(
        "sources",
        "array",
        'List of {"type": "web|news|x"} source configs',
    )
    param_table.add_row(
        "temperature",
        "float",
        "0.0-2.0 - controls response randomness",
    )
    param_table.add_row(
        "max_tokens",
        "integer",
        "Limits response length",
    )
    param_table.add_row(
        "system message",
        "string",
        "Provides context for search interpretation",
    )

    console.print(param_table)

    # Best practices
    console.print("\n[bold yellow]Query Optimization Tips:[/bold yellow]")
    console.print(
        """
    [cyan]1. Be Specific[/cyan]
       Instead of: "Tell me about AI"
       Try: "What are the latest breakthroughs in AI language models in 2024?"

    [cyan]2. Include Time Context[/cyan]
       "Recent news about..." or "Latest developments in..."
       Helps prioritize fresh information

    [cyan]3. Use Domain Context[/cyan]
       Add system messages that establish expertise area
       "You are a [domain] expert..." guides interpretation

    [cyan]4. Control Response Format[/cyan]
       "Provide a bulleted list of..."
       "Summarize in 3 key points..."
       "Give a detailed analysis of..."

    [cyan]5. Specify Source Preferences[/cyan]
       "Focus on official announcements and press releases"
       "Include technical documentation sources"
       "Prioritize peer-reviewed research"

    [cyan]6. Match Sources to Query Type[/cyan]
       - News: Current events, announcements
       - Web: Documentation, tutorials, general info
       - X: Public opinion, real-time reactions
    """
    )


if __name__ == "__main__":
    main()
