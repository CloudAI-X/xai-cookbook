#!/usr/bin/env python3
"""
03_multi_source.py - Combined Multi-Source Search with Grok

This example demonstrates how to combine multiple search sources
(web, news, X/Twitter) in a single query for comprehensive results.

Key concepts:
- Combining multiple source types
- X (Twitter) search with handle filtering
- RSS feed integration
- Source-specific parameters
- Engagement metrics filtering for X
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


def multi_source_search(query: str) -> dict:
    """
    Search across web, news, and X simultaneously.

    Args:
        query: The search query.

    Returns:
        Dictionary containing the combined response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "max_search_results": 15,
                "sources": [
                    {"type": "web"},
                    {"type": "news"},
                    {"type": "x"},  # X (Twitter) posts
                ],
            }
        },
    )

    result = {
        "content": response.choices[0].message.content,
        "sources_used": ["web", "news", "x"],
    }

    if hasattr(response, "citations") and response.citations:
        result["citations"] = response.citations

    return result


def x_search_with_handles(
    query: str,
    included_handles: list[str] | None = None,
    excluded_handles: list[str] | None = None,
) -> dict:
    """
    Search X (Twitter) with handle filtering.

    Args:
        query: The search query.
        included_handles: Only search posts from these handles (max 10).
        excluded_handles: Exclude posts from these handles (max 10).

    Returns:
        Dictionary containing the response.
    """
    x_source = {"type": "x"}

    # Can only use one at a time
    if included_handles:
        x_source["included_x_handles"] = included_handles[:10]
    elif excluded_handles:
        x_source["excluded_x_handles"] = excluded_handles[:10]

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "sources": [x_source],
            }
        },
    )

    return {
        "content": response.choices[0].message.content,
        "handle_filter": (
            f"included: {included_handles}"
            if included_handles
            else f"excluded: {excluded_handles}"
            if excluded_handles
            else "none"
        ),
    }


def x_search_with_engagement(
    query: str,
    min_favorites: int | None = None,
    min_views: int | None = None,
) -> dict:
    """
    Search X with engagement metrics filtering.

    Args:
        query: The search query.
        min_favorites: Minimum number of likes/favorites.
        min_views: Minimum number of views.

    Returns:
        Dictionary containing the response.
    """
    x_source = {"type": "x"}

    if min_favorites:
        x_source["post_favorite_count"] = min_favorites
    if min_views:
        x_source["post_view_count"] = min_views

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "sources": [x_source],
            }
        },
    )

    return {
        "content": response.choices[0].message.content,
        "min_favorites": min_favorites,
        "min_views": min_views,
    }


def rss_feed_search(query: str, rss_url: str) -> dict:
    """
    Search a specific RSS feed.

    Args:
        query: The search query.
        rss_url: URL of the RSS feed to search.

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
                        "type": "rss",
                        "links": [rss_url],  # Single RSS feed URL
                    }
                ],
            }
        },
    )

    return {
        "content": response.choices[0].message.content,
        "rss_feed": rss_url,
    }


def comprehensive_search(query: str, config: dict) -> dict:
    """
    Perform a fully configured multi-source search.

    Args:
        query: The search query.
        config: Configuration dictionary with source settings.

    Returns:
        Dictionary containing the response.
    """
    sources = []

    # Web source
    if config.get("web", {}).get("enabled", True):
        web_source = {"type": "web"}
        if config.get("web", {}).get("country"):
            web_source["country"] = config["web"]["country"]
        sources.append(web_source)

    # News source
    if config.get("news", {}).get("enabled", True):
        news_source = {"type": "news"}
        if config.get("news", {}).get("country"):
            news_source["country"] = config["news"]["country"]
        sources.append(news_source)

    # X source
    if config.get("x", {}).get("enabled", True):
        x_source = {"type": "x"}
        if config.get("x", {}).get("min_favorites"):
            x_source["post_favorite_count"] = config["x"]["min_favorites"]
        sources.append(x_source)

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search_parameters": {
                "mode": "on",
                "return_citations": True,
                "max_search_results": config.get("max_results", 20),
                "from_date": config.get("from_date"),
                "to_date": config.get("to_date"),
                "sources": sources,
            }
        },
    )

    return {
        "content": response.choices[0].message.content,
        "sources_configured": [s["type"] for s in sources],
    }


def main():
    console.print(
        Panel.fit(
            "[bold blue]Multi-Source Search Example[/bold blue]\n"
            "Combine web, news, and X search sources",
            border_style="blue",
        )
    )

    # Example 1: Combined web + news + X search
    console.print("\n[bold yellow]Example 1: Multi-Source Search (Web + News + X)[/bold yellow]")
    console.print("[bold green]Query:[/bold green] What are people saying about AI today?")

    result = multi_source_search(
        "What are people saying about AI today? Give me perspectives from news and social media."
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))
    console.print(f"[dim]Sources used: {', '.join(result['sources_used'])}[/dim]")

    # Example 2: X search with specific handles
    console.print("\n[bold yellow]Example 2: X Search with Specific Handles[/bold yellow]")
    console.print("[bold green]Query:[/bold green] Tech opinions from @elonmusk, @satlogia")

    result = x_search_with_handles(
        "What have they said about technology or AI recently?",
        included_handles=["elonmusk", "satya"],
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))
    console.print(f"[dim]Handle filter: {result['handle_filter']}[/dim]")

    # Example 3: X search with engagement filtering
    console.print("\n[bold yellow]Example 3: X Search with Engagement Filter[/bold yellow]")
    console.print("[bold green]Query:[/bold green] Popular posts about Python (1000+ likes)")

    result = x_search_with_engagement(
        "What are popular posts about Python programming?",
        min_favorites=1000,
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))
    console.print(
        f"[dim]Filters: min_favorites={result['min_favorites']}, "
        f"min_views={result['min_views']}[/dim]"
    )

    # Example 4: Comprehensive configured search
    console.print("\n[bold yellow]Example 4: Comprehensive Configured Search[/bold yellow]")
    console.print("[bold green]Query:[/bold green] SpaceX news and reactions")

    config = {
        "web": {"enabled": True, "country": "US"},
        "news": {"enabled": True, "country": "US"},
        "x": {"enabled": True, "min_favorites": 500},
        "max_results": 15,
    }

    result = comprehensive_search(
        "What's the latest news about SpaceX? Include social media reactions.",
        config,
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(result["content"], border_style="cyan"))
    console.print(f"[dim]Sources: {', '.join(result['sources_configured'])}[/dim]")

    # Source types reference
    console.print("\n[bold yellow]Available Source Types:[/bold yellow]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Source", style="green")
    table.add_column("Description")
    table.add_column("Key Parameters")

    table.add_row(
        "web",
        "General web search",
        "country, allowed_websites, excluded_websites, safe_search",
    )
    table.add_row(
        "news",
        "News publications",
        "country, allowed_websites, excluded_websites, safe_search",
    )
    table.add_row(
        "x",
        "X (Twitter) posts",
        "included_x_handles, excluded_x_handles, post_favorite_count, post_view_count",
    )
    table.add_row("rss", "RSS feed content", "links (single feed URL)")

    console.print(table)

    # X-specific parameters
    console.print("\n[bold yellow]X Search Parameters:[/bold yellow]")

    x_table = Table(show_header=True, header_style="bold magenta")
    x_table.add_column("Parameter", style="green")
    x_table.add_column("Description")
    x_table.add_column("Limit")

    x_table.add_row("included_x_handles", "Only search these accounts", "Max 10")
    x_table.add_row("excluded_x_handles", "Exclude these accounts", "Max 10")
    x_table.add_row("post_favorite_count", "Minimum likes required", "Any integer")
    x_table.add_row("post_view_count", "Minimum views required", "Any integer")

    console.print(x_table)

    # Important notes
    console.print("\n[bold yellow]Important Notes:[/bold yellow]")
    console.print(
        "  - Default sources (if none specified): web, news, x\n"
        "  - Cannot combine allowed_* and excluded_* for same source\n"
        '  - The "grok" handle is auto-excluded from X unless explicitly included\n'
        "  - Each source used counts toward the $25/1000 sources pricing"
    )


if __name__ == "__main__":
    main()
