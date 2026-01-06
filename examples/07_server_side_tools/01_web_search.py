#!/usr/bin/env python3
"""
01_web_search.py - Web Search Server-Side Tool

This example demonstrates how to use xAI's built-in web search capability.
Unlike function calling where you define and execute functions yourself,
server-side tools are executed by xAI's infrastructure automatically.

Web search allows Grok to access real-time information from the internet,
making it possible to answer questions about current events, recent news,
and up-to-date information.

Key concepts:
- Using the extra_body parameter for server-side tools
- Search mode options (auto, on, off)
- Web search source configuration
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


def web_search_auto(client: OpenAI, query: str) -> str:
    """
    Perform a web search with automatic mode.

    In 'auto' mode, Grok decides whether to search based on the query.
    For questions about current events or recent information, it will search.
    For general knowledge questions, it may not.

    Args:
        client: The OpenAI client configured for xAI.
        query: The user's question.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "auto",
                "sources": [{"type": "web"}],
            }
        },
    )

    return response.choices[0].message.content


def web_search_forced(client: OpenAI, query: str) -> str:
    """
    Perform a web search with search always enabled.

    In 'on' mode, Grok will always perform a web search regardless
    of the query type.

    Args:
        client: The OpenAI client configured for xAI.
        query: The user's question.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}],
            }
        },
    )

    return response.choices[0].message.content


def web_search_disabled(client: OpenAI, query: str) -> str:
    """
    Respond without web search capability.

    In 'off' mode, Grok will not perform any web searches and will
    rely solely on its training data.

    Args:
        client: The OpenAI client configured for xAI.
        query: The user's question.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "off",
            }
        },
    )

    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Web Search Server-Side Tool[/bold blue]\n"
            "Using xAI's built-in web search capability",
            border_style="blue",
        )
    )

    client = get_client()

    # Example 1: Auto mode with a current events question
    console.print("\n[bold cyan]Example 1: Auto Mode (Current Events)[/bold cyan]")
    console.print("[dim]Query: 'What are the latest developments in AI this week?'[/dim]")

    query1 = "What are the latest developments in AI this week?"
    response1 = web_search_auto(client, query1)

    console.print(Panel(Markdown(response1), title="Response", border_style="green"))

    # Example 2: Forced web search
    console.print("\n[bold cyan]Example 2: Forced Search Mode[/bold cyan]")
    console.print("[dim]Query: 'What is the current price of Bitcoin?'[/dim]")

    query2 = "What is the current price of Bitcoin?"
    response2 = web_search_forced(client, query2)

    console.print(Panel(Markdown(response2), title="Response", border_style="green"))

    # Example 3: Search disabled for comparison
    console.print("\n[bold cyan]Example 3: Search Disabled[/bold cyan]")
    console.print("[dim]Query: 'Who won the last Super Bowl?' (using training data only)[/dim]")

    query3 = (
        "Who won the last Super Bowl? Note: I understand you may not have the latest information."
    )
    response3 = web_search_disabled(client, query3)

    console.print(Panel(Markdown(response3), title="Response", border_style="yellow"))

    # Show the search configuration options
    console.print("\n[bold yellow]Search Configuration Options:[/bold yellow]")
    console.print(
        """
    [cyan]mode[/cyan]: Controls when search is performed
      - "auto": Grok decides based on the query (default)
      - "on": Always perform search
      - "off": Never perform search

    [cyan]sources[/cyan]: List of search sources to use
      - {"type": "web"}: General web search
      - {"type": "news"}: News-specific search
      - {"type": "x"}: X/Twitter search
    """
    )


if __name__ == "__main__":
    main()
