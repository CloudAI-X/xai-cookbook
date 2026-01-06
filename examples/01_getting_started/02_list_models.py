#!/usr/bin/env python3
"""
02_list_models.py - List Available Models

This script retrieves and displays all available models from the xAI API.
It shows model IDs, ownership, and creation timestamps in a formatted table.

Use this to discover what models are available for your API key.
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

console = Console()


def format_timestamp(unix_timestamp: int) -> str:
    """Convert Unix timestamp to human-readable format."""
    try:
        return datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return "N/A"


def main():
    """List all available models from the xAI API."""
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

    console.print("[bold cyan]Fetching available models from xAI API...[/bold cyan]\n")

    try:
        # List all models
        models_response = client.models.list()
        models = list(models_response)

        if not models:
            console.print("[yellow]No models found.[/yellow]")
            return

        # Create a table for display
        table = Table(title="Available xAI Models", show_header=True, header_style="bold magenta")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Owned By", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Object Type", style="dim")

        # Sort models by ID for consistent display
        sorted_models = sorted(models, key=lambda m: m.id)

        for model in sorted_models:
            table.add_row(
                model.id,
                model.owned_by or "N/A",
                format_timestamp(model.created) if model.created else "N/A",
                model.object or "model",
            )

        console.print(table)
        console.print(f"\n[green]Total models available:[/green] {len(models)}")

        # Group models by type for summary
        model_types = {}
        for model in models:
            # Categorize by prefix
            if "grok" in model.id.lower():
                category = "Grok (Text/Chat)"
            elif "flux" in model.id.lower() or "imagen" in model.id.lower():
                category = "Image Generation"
            elif "embed" in model.id.lower():
                category = "Embeddings"
            else:
                category = "Other"
            model_types.setdefault(category, []).append(model.id)

        if model_types:
            console.print("\n[bold]Models by Category:[/bold]")
            for category, model_ids in sorted(model_types.items()):
                console.print(f"  [cyan]{category}:[/cyan] {len(model_ids)} models")

    except Exception as e:
        console.print(f"[red]Error fetching models:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
