#!/usr/bin/env python3
"""
07_tool_results.py - Handling and Displaying Tool Results

This example demonstrates how to handle and display results from
server-side tools. Understanding the response structure helps you
build better applications that effectively use tool outputs.

Key concepts:
- Understanding response structure with tool use
- Extracting relevant information from responses
- Formatting and displaying tool results
- Error handling for tool operations
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

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


def analyze_response(response) -> dict:
    """
    Analyze the structure of an API response.

    Args:
        response: The API response object.

    Returns:
        Dictionary with analyzed response components.
    """
    analysis = {
        "model": response.model,
        "id": response.id,
        "created": response.created,
        "finish_reason": response.choices[0].finish_reason,
        "content": response.choices[0].message.content,
        "role": response.choices[0].message.role,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": (response.usage.completion_tokens if response.usage else None),
            "total_tokens": response.usage.total_tokens if response.usage else None,
        },
    }

    return analysis


def display_response_tree(analysis: dict) -> None:
    """
    Display response analysis as a tree structure.

    Args:
        analysis: The analyzed response dictionary.
    """
    tree = Tree("[bold]API Response Structure[/bold]")

    # Metadata branch
    meta = tree.add("[cyan]Metadata[/cyan]")
    meta.add(f"ID: {analysis['id']}")
    meta.add(f"Model: {analysis['model']}")
    meta.add(f"Created: {analysis['created']}")
    meta.add(f"Finish Reason: {analysis['finish_reason']}")

    # Message branch
    msg = tree.add("[cyan]Message[/cyan]")
    msg.add(f"Role: {analysis['role']}")
    content_preview = (
        analysis["content"][:100] + "..." if len(analysis["content"]) > 100 else analysis["content"]
    )
    msg.add(f"Content Preview: {content_preview}")

    # Usage branch
    usage = tree.add("[cyan]Token Usage[/cyan]")
    if analysis["usage"]["prompt_tokens"]:
        usage.add(f"Prompt Tokens: {analysis['usage']['prompt_tokens']}")
        usage.add(f"Completion Tokens: {analysis['usage']['completion_tokens']}")
        usage.add(f"Total Tokens: {analysis['usage']['total_tokens']}")
    else:
        usage.add("Usage data not available")

    console.print(tree)


def search_with_detailed_response(client: OpenAI, query: str) -> tuple:
    """
    Perform a search and return both content and metadata.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        Tuple of (content, analysis).
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": query}],
        extra_body={
            "search": {
                "mode": "on",
                "sources": [{"type": "web"}, {"type": "news"}],
            }
        },
    )

    analysis = analyze_response(response)
    return response.choices[0].message.content, analysis


def format_search_results(content: str, format_type: str = "markdown") -> str:
    """
    Format search results for different output types.

    Args:
        content: The raw content from the API.
        format_type: The output format (markdown, plain, summary).

    Returns:
        Formatted content string.
    """
    if format_type == "plain":
        # Strip markdown formatting for plain text
        import re

        # Remove markdown headers
        plain = re.sub(r"#{1,6}\s+", "", content)
        # Remove bold/italic
        plain = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", plain)
        # Remove links, keep text
        plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", plain)
        return plain

    elif format_type == "summary":
        # Return just the first paragraph or section
        paragraphs = content.split("\n\n")
        return paragraphs[0] if paragraphs else content

    else:  # markdown
        return content


def handle_search_errors(client: OpenAI, query: str) -> str:
    """
    Demonstrate error handling for search operations.

    Args:
        client: The OpenAI client configured for xAI.
        query: The search query.

    Returns:
        The response content or error message.
    """
    try:
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

        if not response.choices:
            return "Error: No response choices returned"

        if not response.choices[0].message.content:
            return "Error: Empty response content"

        return response.choices[0].message.content

    except Exception as e:
        return f"Error during search: {str(e)}"


def main():
    console.print(
        Panel.fit(
            "[bold blue]Handling and Displaying Tool Results[/bold blue]\n"
            "Understanding and working with API responses",
            border_style="blue",
        )
    )

    client = get_client()

    # Example 1: Analyze response structure
    console.print("\n[bold cyan]Example 1: Response Structure Analysis[/bold cyan]")
    console.print("[dim]Query: 'What is the latest news about renewable energy?'[/dim]")

    content, analysis = search_with_detailed_response(
        client, "What is the latest news about renewable energy?"
    )

    display_response_tree(analysis)

    # Example 2: Display usage statistics
    console.print("\n[bold cyan]Example 2: Token Usage Statistics[/bold cyan]")

    usage_table = Table(title="Token Usage")
    usage_table.add_column("Metric", style="cyan")
    usage_table.add_column("Value", style="green")

    if analysis["usage"]["prompt_tokens"]:
        usage_table.add_row("Prompt Tokens", str(analysis["usage"]["prompt_tokens"]))
        usage_table.add_row("Completion Tokens", str(analysis["usage"]["completion_tokens"]))
        usage_table.add_row("Total Tokens", str(analysis["usage"]["total_tokens"]))

        # Calculate rough cost estimation (example rates)
        prompt_cost = analysis["usage"]["prompt_tokens"] * 0.000001
        completion_cost = analysis["usage"]["completion_tokens"] * 0.000002
        usage_table.add_row("Estimated Cost", f"${prompt_cost + completion_cost:.6f}")

    console.print(usage_table)

    # Example 3: Different output formats
    console.print("\n[bold cyan]Example 3: Output Format Options[/bold cyan]")

    console.print("\n[bold green]Markdown Format:[/bold green]")
    console.print(
        Panel(
            Markdown(format_search_results(content, "markdown")[:500] + "..."),
            border_style="green",
        )
    )

    console.print("\n[bold green]Plain Text Format:[/bold green]")
    console.print(
        Panel(format_search_results(content, "plain")[:500] + "...", border_style="yellow")
    )

    console.print("\n[bold green]Summary Format:[/bold green]")
    console.print(Panel(format_search_results(content, "summary"), border_style="cyan"))

    # Example 4: Error handling
    console.print("\n[bold cyan]Example 4: Error Handling[/bold cyan]")

    result = handle_search_errors(client, "What's happening in technology today?")

    if result.startswith("Error"):
        console.print(f"[red]{result}[/red]")
    else:
        console.print("[green]Search completed successfully[/green]")
        console.print(Panel(Markdown(result[:300] + "..."), border_style="green"))

    # Show response handling patterns
    console.print("\n[bold yellow]Response Handling Patterns:[/bold yellow]")
    console.print(
        """
    [cyan]Accessing Response Data:[/cyan]

    # Content
    content = response.choices[0].message.content

    # Finish reason
    finish_reason = response.choices[0].finish_reason

    # Token usage
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    [cyan]Error Handling:[/cyan]

    try:
        response = client.chat.completions.create(...)
        if not response.choices:
            handle_empty_response()
        if response.choices[0].finish_reason == "length":
            handle_truncated_response()
    except openai.APIError as e:
        handle_api_error(e)
    except openai.RateLimitError as e:
        handle_rate_limit(e)

    [cyan]Finish Reasons:[/cyan]
    - "stop": Normal completion
    - "length": Max tokens reached
    - "tool_calls": Model wants to call tools
    - "content_filter": Content was filtered
    """
    )


if __name__ == "__main__":
    main()
