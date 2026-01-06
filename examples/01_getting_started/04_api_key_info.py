#!/usr/bin/env python3
"""
04_api_key_info.py - Check API Key Information

This script checks your xAI API key status by making a simple API call.
It verifies that your API key is valid and shows basic account information.

Note: The xAI API (via OpenAI SDK) doesn't have a dedicated endpoint for
API key info, so we verify the key by listing models and making a test call.
"""

import os
import sys

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    """Check API key validity and display account information."""
    # Load environment variables from .env file
    load_dotenv()

    # Check for API key
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        console.print(
            Panel(
                "[red]X_AI_API_KEY environment variable not set![/red]\n\n"
                "To get started:\n"
                "1. Visit https://console.x.ai to create an API key\n"
                "2. Create a .env file with: X_AI_API_KEY=your-key-here\n"
                "3. Or set it in your environment: export X_AI_API_KEY=your-key",
                title="API Key Missing",
                border_style="red",
            )
        )
        sys.exit(1)

    # Display masked API key
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    console.print(f"[bold cyan]Checking API key:[/bold cyan] {masked_key}\n")

    # Initialize the client with xAI's base URL
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    try:
        # Test 1: List models to verify API access
        console.print("[dim]Testing API access...[/dim]")
        models = list(client.models.list())
        model_count = len(models)

        # Test 2: Make a minimal completion request
        console.print("[dim]Testing chat completion...[/dim]")
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        # All tests passed - display success
        console.print()

        # Create status table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Check", style="bold")
        table.add_column("Status", style="white")

        table.add_row("API Key Valid", "[green]Yes[/green]")
        table.add_row("Models Accessible", f"[green]{model_count} models available[/green]")
        table.add_row("Chat Completions", "[green]Working[/green]")
        table.add_row("API Endpoint", "https://api.x.ai/v1")

        console.print(
            Panel(
                table,
                title="[bold green]API Key Status: Valid[/bold green]",
                border_style="green",
            )
        )

        # Show available model categories
        console.print("\n[bold]Available Model Categories:[/bold]")
        categories = {
            "Text/Chat": [
                m.id for m in models if "grok" in m.id.lower() and "vision" not in m.id.lower()
            ],
            "Vision": [m.id for m in models if "vision" in m.id.lower()],
            "Image Gen": [
                m.id for m in models if "flux" in m.id.lower() or "imagen" in m.id.lower()
            ],
            "Embeddings": [m.id for m in models if "embed" in m.id.lower()],
        }

        for category, model_ids in categories.items():
            if model_ids:
                console.print(f"  [cyan]{category}:[/cyan] {', '.join(sorted(model_ids)[:3])}")
                if len(model_ids) > 3:
                    console.print(f"    [dim]...and {len(model_ids) - 3} more[/dim]")

        console.print("\n[green]Your API key is ready to use![/green]")

    except AuthenticationError:
        console.print(
            Panel(
                "[red]API key is invalid or expired![/red]\n\n"
                "Please check your API key:\n"
                "1. Verify the key at https://console.x.ai\n"
                "2. Make sure you copied the entire key\n"
                "3. Check that the key hasn't been revoked",
                title="Authentication Failed",
                border_style="red",
            )
        )
        sys.exit(1)

    except Exception as e:
        error_message = str(e)
        console.print(
            Panel(
                f"[red]Error checking API key:[/red]\n\n{error_message}\n\n"
                "This could be due to:\n"
                "- Network connectivity issues\n"
                "- API service temporarily unavailable\n"
                "- Rate limiting",
                title="API Error",
                border_style="red",
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
