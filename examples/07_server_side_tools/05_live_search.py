#!/usr/bin/env python3
"""
05_live_search.py - Live Search with Multiple Sources

This example demonstrates how to use xAI's live search capability
with multiple sources simultaneously. Live search combines results
from web, news, and X to provide comprehensive, real-time information.

Live search is useful for:
- Getting comprehensive information on current events
- Combining multiple perspectives (news, social media, web)
- Real-time information gathering

Key concepts:
- Combining multiple search sources
- News-specific search configuration
- Understanding source priorities
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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


def live_search_all_sources(client: OpenAI, query: str) -> str:
    """
    Perform a live search across all available sources.

    Combines web, news, and X search for comprehensive results.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        The assistant's response with combined search results.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [
                    {"type": "web"},
                    {"type": "x"},
                    {"type": "news"},
                ],
            }
        },
    )

    return response.choices[0].message.content


def news_search(client: OpenAI, query: str) -> str:
    """
    Perform a news-specific search.

    Focuses on news sources for current events and reporting.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        The assistant's response with news search results.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "news"}],
            }
        },
    )

    return response.choices[0].message.content


def web_and_news_search(client: OpenAI, query: str) -> str:
    """
    Combine web and news search.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        The assistant's response with combined results.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [
                    {"type": "web"},
                    {"type": "news"},
                ],
            }
        },
    )

    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Live Search with Multiple Sources[/bold blue]\n"
            "Combine web, news, and X search for comprehensive results",
            border_style="blue",
        )
    )

    client = get_client()

    # Example 1: All sources for breaking news
    console.print("\n[bold cyan]Example 1: All Sources Search (Breaking News)[/bold cyan]")
    console.print("[dim]Query: 'What are the major tech news stories today?'[/dim]")

    query1 = "What are the major tech news stories today?"
    response1 = live_search_all_sources(client, query1)

    console.print(Panel(Markdown(response1), title="All Sources Results", border_style="green"))

    # Example 2: News-specific search
    console.print("\n[bold cyan]Example 2: News-Only Search[/bold cyan]")
    console.print("[dim]Query: 'Latest news about climate change policy developments'[/dim]")

    query2 = "Latest news about climate change policy developments"
    response2 = news_search(client, query2)

    console.print(Panel(Markdown(response2), title="News Search Results", border_style="green"))

    # Example 3: Web + News combined
    console.print("\n[bold cyan]Example 3: Web + News Search[/bold cyan]")
    console.print("[dim]Query: 'What's happening with electric vehicle adoption?'[/dim]")

    query3 = "What's happening with electric vehicle adoption?"
    response3 = web_and_news_search(client, query3)

    console.print(Panel(Markdown(response3), title="Web + News Results", border_style="green"))

    # Example 4: Multi-source for complex topics
    console.print("\n[bold cyan]Example 4: Multi-Source Complex Query[/bold cyan]")
    console.print(
        "[dim]Query: 'What is the current state of space exploration? Include recent launches, "
        "public discussions, and news coverage.'[/dim]"
    )

    query4 = (
        "What is the current state of space exploration? Include recent launches, "
        "public discussions, and news coverage."
    )
    response4 = live_search_all_sources(client, query4)

    console.print(Panel(Markdown(response4), title="Comprehensive Results", border_style="green"))

    # Show source configuration options
    console.print("\n[bold yellow]Search Source Configuration:[/bold yellow]")
    console.print(
        """
    [cyan]Available Sources:[/cyan]

    {"type": "web"}
      - General web search
      - Best for: documentation, articles, general information

    {"type": "news"}
      - News-specific search
      - Best for: current events, breaking news, journalism

    {"type": "x"}
      - X/Twitter search
      - Best for: real-time reactions, public sentiment, discussions

    [cyan]Combining Sources:[/cyan]

    "sources": [
        {"type": "web"},
        {"type": "news"},
        {"type": "x"}
    ]

    Results are synthesized from all specified sources into
    a coherent response.

    [cyan]Best Practices:[/cyan]
    - Use all sources for comprehensive coverage
    - Use news-only for factual reporting
    - Use X-only for public opinion and sentiment
    - Use web-only for technical/documentation queries
    """
    )


if __name__ == "__main__":
    main()
