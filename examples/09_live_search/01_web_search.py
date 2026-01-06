#!/usr/bin/env python3
"""
01_web_search.py - Real-time Web Search with Grok

This example demonstrates how to use xAI's live search feature to get
real-time information from the web. Live search allows Grok to access
current information beyond its training data.

Key concepts:
- Enabling live search with search_parameters
- Web source configuration
- Handling citations and sources
- Date filtering for searches
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


def web_search_basic(query: str) -> dict:
    """
    Perform a basic web search with live data.

    Args:
        query: The search query to send to Grok.

    Returns:
        Dictionary containing the response and search metadata.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        # Enable live search with web sources
        extra_body={
            "search_parameters": {
                "mode": "on",  # Force live search (options: "off", "auto", "on")
                "return_citations": True,
                "sources": [
                    {
                        "type": "web",
                        "country": "US",  # ISO alpha-2 country code
                    }
                ],
            }
        },
    )

    result = {
        "content": response.choices[0].message.content,
        "model": response.model,
        "citations": [],
    }

    # Extract citations if available
    if hasattr(response, "citations") and response.citations:
        result["citations"] = response.citations

    # Check usage for sources used
    if response.usage:
        result["prompt_tokens"] = response.usage.prompt_tokens
        result["completion_tokens"] = response.usage.completion_tokens
        # num_sources_used may be available in extra fields
        if hasattr(response.usage, "num_sources_used"):
            result["sources_used"] = response.usage.num_sources_used

    return result


def web_search_with_filters(
    query: str,
    allowed_websites: list[str] | None = None,
    excluded_websites: list[str] | None = None,
) -> dict:
    """
    Perform a web search with domain filtering.

    Args:
        query: The search query.
        allowed_websites: Only search these domains (max 5).
        excluded_websites: Exclude these domains from search (max 5).

    Returns:
        Dictionary containing the response and metadata.
    """
    web_source = {"type": "web"}

    # Add filters (can only use one at a time)
    if allowed_websites:
        web_source["allowed_websites"] = allowed_websites[:5]
    elif excluded_websites:
        web_source["excluded_websites"] = excluded_websites[:5]

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "max_search_results": 10,  # Limit sources considered
                "sources": [web_source],
            }
        },
    )

    return {
        "content": response.choices[0].message.content,
        "filter_type": (
            "allowed" if allowed_websites else "excluded" if excluded_websites else None
        ),
    }


def web_search_with_date_range(
    query: str, from_date: str | None = None, to_date: str | None = None
) -> dict:
    """
    Perform a web search with date range filtering.

    Args:
        query: The search query.
        from_date: Start date in ISO8601 format (YYYY-MM-DD).
        to_date: End date in ISO8601 format (YYYY-MM-DD).

    Returns:
        Dictionary containing the response.
    """
    search_params = {
        "mode": "on",
        "return_citations": True,
        "sources": [{"type": "web"}],
    }

    if from_date:
        search_params["from_date"] = from_date
    if to_date:
        search_params["to_date"] = to_date

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={"search_parameters": search_params},
    )

    return {
        "content": response.choices[0].message.content,
        "date_range": f"{from_date or 'any'} to {to_date or 'now'}",
    }


def main():
    console.print(
        Panel.fit(
            "[bold blue]Live Web Search Example[/bold blue]\n"
            "Get real-time information from the web",
            border_style="blue",
        )
    )

    # Example 1: Basic web search
    console.print("\n[bold yellow]Example 1: Basic Web Search[/bold yellow]")
    console.print("[bold green]Query:[/bold green] What are the latest tech news today?")

    result = web_search_basic("What are the latest tech news today? Give me 3 headlines.")

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))

    if result.get("citations"):
        console.print("\n[bold magenta]Citations:[/bold magenta]")
        for i, citation in enumerate(result["citations"][:5], 1):
            console.print(f"  {i}. {citation}")

    # Example 2: Search with domain filtering
    console.print("\n[bold yellow]Example 2: Domain-Filtered Search[/bold yellow]")
    console.print(
        "[bold green]Query:[/bold green] Latest AI developments "
        "(only from techcrunch.com, wired.com)"
    )

    result = web_search_with_filters(
        "What are the latest AI developments?",
        allowed_websites=["techcrunch.com", "wired.com"],
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))

    # Example 3: Search with exclusions
    console.print("\n[bold yellow]Example 3: Search with Exclusions[/bold yellow]")
    console.print(
        "[bold green]Query:[/bold green] Python programming tips "
        "(excluding reddit.com, stackoverflow.com)"
    )

    result = web_search_with_filters(
        "Give me 3 Python programming tips for beginners.",
        excluded_websites=["reddit.com", "stackoverflow.com"],
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))

    # Example 4: Date-filtered search
    console.print("\n[bold yellow]Example 4: Date-Filtered Search[/bold yellow]")
    console.print("[bold green]Query:[/bold green] AI news from December 2024")

    result = web_search_with_date_range(
        "What were the major AI announcements?",
        from_date="2024-12-01",
        to_date="2024-12-31",
    )

    console.print(f"\n[dim]Date range: {result['date_range']}[/dim]")
    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))

    # Show search parameters reference
    console.print("\n[bold yellow]Search Parameters Reference:[/bold yellow]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="green")
    table.add_column("Description")
    table.add_column("Default")

    table.add_row("mode", '"off", "auto", or "on"', '"auto"')
    table.add_row("return_citations", "Return source URLs", "true")
    table.add_row("max_search_results", "Max sources to consider", "20")
    table.add_row("from_date", "Start date (ISO8601)", "None")
    table.add_row("to_date", "End date (ISO8601)", "None")

    console.print(table)

    # Pricing note
    console.print("\n[bold yellow]Pricing Note:[/bold yellow]")
    console.print(
        "  Live search costs [cyan]$25 per 1,000 sources[/cyan] ($0.025/source)\n"
        "  Source count available in response.usage.num_sources_used"
    )


if __name__ == "__main__":
    main()
