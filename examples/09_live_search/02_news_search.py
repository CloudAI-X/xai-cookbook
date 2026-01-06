#!/usr/bin/env python3
"""
02_news_search.py - News Source Search with Grok

This example demonstrates how to search news sources specifically using
xAI's live search feature. News search provides curated results from
news outlets and publications.

Key concepts:
- Configuring news-specific sources
- Country-based news filtering
- Safe search options
- Combining with date ranges for recent news
"""

import os
from datetime import datetime, timedelta

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


def news_search(
    query: str,
    country: str = "US",
    safe_search: bool = True,
    max_results: int = 10,
) -> dict:
    """
    Search news sources for current events.

    Args:
        query: The news topic to search for.
        country: ISO alpha-2 country code for regional news.
        safe_search: Enable safe search filtering.
        max_results: Maximum number of sources to consider.

    Returns:
        Dictionary containing the response and metadata.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "max_search_results": max_results,
                "sources": [
                    {
                        "type": "news",
                        "country": country,
                        "safe_search": safe_search,
                    }
                ],
            }
        },
    )

    result = {
        "content": response.choices[0].message.content,
        "country": country,
    }

    if hasattr(response, "citations") and response.citations:
        result["citations"] = response.citations

    return result


def news_search_with_exclusions(
    query: str,
    excluded_sources: list[str],
) -> dict:
    """
    Search news with specific sources excluded.

    Args:
        query: The news topic to search for.
        excluded_sources: List of domains to exclude (max 5).

    Returns:
        Dictionary containing the response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "sources": [
                    {
                        "type": "news",
                        "excluded_websites": excluded_sources[:5],
                    }
                ],
            }
        },
    )

    return {
        "content": response.choices[0].message.content,
        "excluded": excluded_sources[:5],
    }


def recent_news_search(query: str, days_back: int = 7) -> dict:
    """
    Search for news from the last N days.

    Args:
        query: The news topic to search for.
        days_back: How many days back to search.

    Returns:
        Dictionary containing the response and date range.
    """
    today = datetime.now()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "from_date": from_date,
                "to_date": to_date,
                "sources": [{"type": "news"}],
            }
        },
    )

    return {
        "content": response.choices[0].message.content,
        "from_date": from_date,
        "to_date": to_date,
        "days_back": days_back,
    }


def international_news_comparison(query: str, countries: list[str]) -> list[dict]:
    """
    Compare news coverage across different countries.

    Args:
        query: The news topic to search for.
        countries: List of ISO alpha-2 country codes.

    Returns:
        List of results from each country.
    """
    results = []

    for country in countries:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {
                    "role": "user",
                    "content": f"{query} Summarize in 2-3 sentences.",
                }
            ],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                    "return_citations": True,
                    "max_search_results": 5,
                    "sources": [{"type": "news", "country": country}],
                }
            },
        )

        results.append(
            {
                "country": country,
                "content": response.choices[0].message.content,
            }
        )

    return results


def main():
    console.print(
        Panel.fit(
            "[bold blue]News Search Example[/bold blue]\nSearch news sources for current events",
            border_style="blue",
        )
    )

    # Example 1: Basic news search
    console.print("\n[bold yellow]Example 1: Basic News Search[/bold yellow]")
    console.print("[bold green]Query:[/bold green] What are today's top business headlines?")

    result = news_search(
        "What are today's top business headlines? List 3 stories.",
        country="US",
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))

    if result.get("citations"):
        console.print("\n[bold magenta]News Sources:[/bold magenta]")
        for i, citation in enumerate(result["citations"][:5], 1):
            console.print(f"  {i}. {citation}")

    # Example 2: News with exclusions
    console.print("\n[bold yellow]Example 2: News with Source Exclusions[/bold yellow]")
    console.print(
        "[bold green]Query:[/bold green] Technology news (excluding cnn.com, foxnews.com)"
    )

    result = news_search_with_exclusions(
        "What's happening in the technology industry today? Give me 2 stories.",
        excluded_sources=["cnn.com", "foxnews.com"],
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))
    console.print(f"[dim]Excluded sources: {', '.join(result['excluded'])}[/dim]")

    # Example 3: Recent news search
    console.print("\n[bold yellow]Example 3: Recent News (Last 7 Days)[/bold yellow]")
    console.print("[bold green]Query:[/bold green] AI breakthroughs this week")

    result = recent_news_search(
        "What AI breakthroughs or announcements happened? List the top 3.",
        days_back=7,
    )

    console.print(f"\n[dim]Date range: {result['from_date']} to {result['to_date']}[/dim]")
    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))

    # Example 4: International news comparison
    console.print("\n[bold yellow]Example 4: International News Comparison[/bold yellow]")
    console.print("[bold green]Query:[/bold green] Climate news from US, UK, DE")

    results = international_news_comparison(
        "What's the latest climate or environmental news?",
        countries=["US", "GB", "DE"],
    )

    for result in results:
        country_names = {"US": "United States", "GB": "United Kingdom", "DE": "Germany"}
        name = country_names.get(result["country"], result["country"])
        console.print(f"\n[bold magenta]{name} ({result['country']}):[/bold magenta]")
        console.print(Panel(result["content"], border_style="magenta"))

    # News source configuration reference
    console.print("\n[bold yellow]News Source Configuration:[/bold yellow]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="green")
    table.add_column("Description")
    table.add_column("Example")

    table.add_row("type", "Source type", '"news"')
    table.add_row("country", "ISO alpha-2 country code", '"US", "GB", "DE"')
    table.add_row("safe_search", "Filter inappropriate content", "true/false")
    table.add_row("excluded_websites", "Domains to exclude (max 5)", '["cnn.com"]')
    table.add_row("allowed_websites", "Only search these (max 5)", '["bbc.com"]')

    console.print(table)

    # Country codes reference
    console.print("\n[bold yellow]Common Country Codes:[/bold yellow]")
    console.print(
        "  US (United States), GB (United Kingdom), CA (Canada), AU (Australia)\n"
        "  DE (Germany), FR (France), JP (Japan), IN (India), BR (Brazil)"
    )


if __name__ == "__main__":
    main()
