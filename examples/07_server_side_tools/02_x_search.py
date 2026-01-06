#!/usr/bin/env python3
"""
02_x_search.py - X/Twitter Search Server-Side Tool

This example demonstrates how to use xAI's X (formerly Twitter) search
capability. This allows Grok to search through posts, users, and trending
topics on the X platform.

X search is particularly useful for:
- Finding real-time reactions to events
- Searching for expert opinions
- Discovering trending topics
- Finding posts from specific users or about specific topics

Key concepts:
- Configuring X as a search source
- Searching for posts and discussions
- Combining X search with natural language queries
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


def search_x_posts(client: OpenAI, query: str) -> str:
    """
    Search X/Twitter for posts related to a query.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        The assistant's response with X search results.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "x"}],
            }
        },
    )

    return response.choices[0].message.content


def search_x_auto(client: OpenAI, query: str) -> str:
    """
    Search X/Twitter in auto mode.

    In auto mode, Grok decides whether to search X based on the query.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "auto",
                "sources": [{"type": "x"}],
            }
        },
    )

    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]X/Twitter Search Server-Side Tool[/bold blue]\n"
            "Search posts and discussions on X",
            border_style="blue",
        )
    )

    client = get_client()

    # Example 1: Search for trending discussions
    console.print("\n[bold cyan]Example 1: Trending Discussions[/bold cyan]")
    console.print(
        "[dim]Query: 'What are people on X saying about artificial intelligence today?'[/dim]"
    )

    query1 = "What are people on X saying about artificial intelligence today?"
    response1 = search_x_posts(client, query1)

    console.print(Panel(Markdown(response1), title="X Search Results", border_style="green"))

    # Example 2: Search for specific topic reactions
    console.print("\n[bold cyan]Example 2: Topic-Specific Search[/bold cyan]")
    console.print("[dim]Query: 'Find recent posts on X about SpaceX launches'[/dim]")

    query2 = "Find recent posts on X about SpaceX launches"
    response2 = search_x_posts(client, query2)

    console.print(Panel(Markdown(response2), title="X Search Results", border_style="green"))

    # Example 3: Auto mode for contextual search
    console.print("\n[bold cyan]Example 3: Auto Mode Search[/bold cyan]")
    console.print("[dim]Query: 'What's the public sentiment on X about electric vehicles?'[/dim]")

    query3 = "What's the public sentiment on X about electric vehicles?"
    response3 = search_x_auto(client, query3)

    console.print(Panel(Markdown(response3), title="X Search Results", border_style="green"))

    # Show use cases for X search
    console.print("\n[bold yellow]X Search Use Cases:[/bold yellow]")
    console.print(
        """
    [cyan]Real-time Reactions[/cyan]
      Search for live reactions to events, announcements, or news

    [cyan]Public Sentiment[/cyan]
      Gauge public opinion on topics, products, or companies

    [cyan]Expert Opinions[/cyan]
      Find posts from industry experts and thought leaders

    [cyan]Trending Topics[/cyan]
      Discover what's currently being discussed

    [cyan]Community Discussions[/cyan]
      Find conversations in specific communities or niches
    """
    )

    # Show configuration
    console.print("\n[bold yellow]X Search Configuration:[/bold yellow]")
    console.print(
        """
    extra_body={
        "search": {
            "mode": "on",  # or "auto"
            "sources": [{"type": "x"}]
        }
    }
    """
    )


if __name__ == "__main__":
    main()
