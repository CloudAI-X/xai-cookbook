#!/usr/bin/env python3
"""
01_hello_grok.py - Hello World with Grok

This script demonstrates the simplest possible interaction with the xAI API.
It sends a "Hello World" message to grok-4-1-fast-reasoning and displays the response.

This is the perfect starting point for your xAI journey!
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    """Send a Hello World message to Grok and display the response."""
    # Load environment variables from .env file
    load_dotenv()

    # Check for API key
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] X_AI_API_KEY environment variable not set.\n"
            "Please set it in your .env file or environment."
        )
        sys.exit(1)

    # Initialize the client with xAI's base URL
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    console.print(
        Panel(
            "[bold cyan]Hello Grok![/bold cyan]\n\n"
            "Sending your first message to grok-4-1-fast-reasoning...",
            title="xAI Getting Started",
            border_style="cyan",
        )
    )

    try:
        # Create a simple chat completion
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {
                    "role": "user",
                    "content": "Hello! Please introduce yourself in one sentence.",
                }
            ],
        )

        # Extract and display the response
        assistant_message = response.choices[0].message.content

        console.print("\n[bold green]Grok's Response:[/bold green]")
        console.print(Panel(assistant_message, border_style="green"))

        # Show some metadata
        console.print("\n[dim]Model used:[/dim]", response.model)
        if response.usage:
            console.print(
                f"[dim]Tokens - Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error communicating with xAI API:[/red] {e}")
        sys.exit(1)

    console.print("\n[green]Success![/green] You've made your first xAI API call.")


if __name__ == "__main__":
    main()
