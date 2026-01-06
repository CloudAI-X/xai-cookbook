#!/usr/bin/env python3
"""
06_combined_tools.py - Using Multiple Server-Side Tools Together

This example demonstrates how to combine multiple xAI server-side tools
in a single request or conversation. Combining tools allows for more
comprehensive and capable interactions.

Combined tool use is useful for:
- Complex queries that need multiple information sources
- Tasks requiring both search and computation
- Building comprehensive research workflows

Key concepts:
- Combining search with other capabilities
- Multi-turn conversations with tool context
- Building research workflows
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


def research_with_multi_source(client: OpenAI, topic: str) -> str:
    """
    Perform comprehensive research using multiple search sources.

    Args:
        client: The OpenAI client configured for xAI.
        topic: The research topic.

    Returns:
        The assistant's comprehensive research response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. When given a topic, provide "
                    "a comprehensive overview including recent news, public "
                    "discussions, and general information. Organize your response "
                    "with clear sections."
                ),
            },
            {"role": "user", "content": f"Research topic: {topic}"},
        ],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [
                    {"type": "web"},
                    {"type": "news"},
                    {"type": "x"},
                ],
            }
        },
    )

    return response.choices[0].message.content


def multi_turn_research(client: OpenAI, initial_query: str, follow_up: str) -> tuple:
    """
    Demonstrate multi-turn conversation with search tools.

    Args:
        client: The OpenAI client configured for xAI.
        initial_query: The first research question.
        follow_up: A follow-up question.

    Returns:
        Tuple of (initial_response, follow_up_response).
    """
    # First turn - initial research
    messages = [
        {
            "role": "system",
            "content": "You are a helpful research assistant with access to real-time information.",
        },
        {"role": "user", "content": initial_query},
    ]

    response1 = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=messages,
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}, {"type": "news"}],
            }
        },
    )

    initial_response = response1.choices[0].message.content

    # Second turn - follow-up with context
    messages.append({"role": "assistant", "content": initial_response})
    messages.append({"role": "user", "content": follow_up})

    response2 = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=messages,
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}, {"type": "x"}],
            }
        },
    )

    follow_up_response = response2.choices[0].message.content

    return initial_response, follow_up_response


def compare_perspectives(client: OpenAI, topic: str) -> dict:
    """
    Compare different perspectives on a topic using different sources.

    Args:
        client: The OpenAI client configured for xAI.
        topic: The topic to analyze.

    Returns:
        Dictionary with perspectives from different sources.
    """
    results = {}

    # News perspective
    news_response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "user",
                "content": f"What do news sources say about: {topic}",
            }
        ],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "news"}],
            }
        },
    )
    results["news"] = news_response.choices[0].message.content

    # Social media perspective
    social_response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "user",
                "content": f"What are people on X saying about: {topic}",
            }
        ],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "x"}],
            }
        },
    )
    results["social"] = social_response.choices[0].message.content

    return results


def main():
    console.print(
        Panel.fit(
            "[bold blue]Combined Server-Side Tools[/bold blue]\n"
            "Using multiple tools together for comprehensive results",
            border_style="blue",
        )
    )

    client = get_client()

    # Example 1: Comprehensive research
    console.print("\n[bold cyan]Example 1: Comprehensive Multi-Source Research[/bold cyan]")
    console.print("[dim]Topic: 'Artificial Intelligence in Healthcare'[/dim]")

    research_topic = "Artificial Intelligence in Healthcare - recent developments and applications"
    research_result = research_with_multi_source(client, research_topic)

    console.print(Panel(Markdown(research_result), title="Research Results", border_style="green"))

    # Example 2: Multi-turn research conversation
    console.print("\n[bold cyan]Example 2: Multi-Turn Research Conversation[/bold cyan]")
    console.print("[dim]Initial: 'What is quantum computing?'[/dim]")
    console.print(
        "[dim]Follow-up: 'What are people saying about its practical applications?'[/dim]"
    )

    initial_resp, followup_resp = multi_turn_research(
        client,
        "What is quantum computing and what are the latest breakthroughs?",
        "What are people on X saying about its practical applications?",
    )

    console.print(Panel(Markdown(initial_resp), title="Initial Response", border_style="green"))
    console.print(Panel(Markdown(followup_resp), title="Follow-up Response", border_style="blue"))

    # Example 3: Compare perspectives
    console.print("\n[bold cyan]Example 3: Compare Perspectives Across Sources[/bold cyan]")
    console.print("[dim]Topic: 'Renewable Energy'[/dim]")

    perspectives = compare_perspectives(client, "Renewable energy transition")

    console.print(
        Panel(
            Markdown(perspectives["news"]),
            title="News Perspective",
            border_style="green",
        )
    )
    console.print(
        Panel(
            Markdown(perspectives["social"]),
            title="Social Media Perspective",
            border_style="cyan",
        )
    )

    # Show combined tool patterns
    console.print("\n[bold yellow]Combined Tool Patterns:[/bold yellow]")
    console.print(
        """
    [cyan]Multi-Source Research[/cyan]
      Combine web, news, and X for comprehensive coverage:

      extra_body={
          "search": {
              "mode": "on",
              "sources": [
                  {"type": "web"},
                  {"type": "news"},
                  {"type": "x"}
              ]
          }
      }

    [cyan]Multi-Turn Conversations[/cyan]
      Maintain context across turns while using different sources:
      - Turn 1: Use news for factual information
      - Turn 2: Use X for public sentiment
      - Turn 3: Use web for technical details

    [cyan]Perspective Comparison[/cyan]
      Query the same topic with different sources to compare:
      - News: Professional journalism perspective
      - X: Public opinion and sentiment
      - Web: Technical and detailed information

    [cyan]Best Practices[/cyan]
      - Use system prompts to guide response organization
      - Build conversation history for context-aware follow-ups
      - Match source types to information needs
    """
    )


if __name__ == "__main__":
    main()
