#!/usr/bin/env python3
"""
03_model_info.py - Get Model Information

This script retrieves detailed information about a specific xAI model.
You can pass a model ID as a command-line argument or use the default.

Usage:
    python 03_model_info.py                  # Shows info for grok-4-1-fast-reasoning
    python 03_model_info.py grok-3           # Shows info for grok-3
    python 03_model_info.py grok-vision-beta # Shows info for vision model
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import NotFoundError, OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Default model to query if none specified
DEFAULT_MODEL = "grok-4-1-fast-reasoning"


def format_timestamp(unix_timestamp: int) -> str:
    """Convert Unix timestamp to human-readable format."""
    try:
        return datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return "N/A"


def main():
    """Get detailed information about a specific model."""
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

    # Get model ID from command line or use default
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    # Initialize the client with xAI's base URL
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    console.print(f"[bold cyan]Fetching information for model:[/bold cyan] {model_id}\n")

    try:
        # Retrieve model information
        model = client.models.retrieve(model_id)

        # Create information table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Model ID", model.id)
        table.add_row("Object Type", model.object or "model")
        table.add_row("Owned By", model.owned_by or "N/A")
        table.add_row("Created", format_timestamp(model.created) if model.created else "N/A")

        # Display the information
        console.print(
            Panel(
                table,
                title=f"[bold]Model: {model.id}[/bold]",
                border_style="green",
            )
        )

        # Model capabilities hints based on model name
        capabilities = []
        model_lower = model.id.lower()

        if "vision" in model_lower:
            capabilities.append("Vision/Image Understanding")
        if "mini" in model_lower:
            capabilities.append("Fast, Cost-Effective")
        if "grok-3" in model_lower and "mini" not in model_lower:
            capabilities.append("High Intelligence")
        if "embed" in model_lower:
            capabilities.append("Text Embeddings")
        if "flux" in model_lower or "imagen" in model_lower:
            capabilities.append("Image Generation")

        if capabilities:
            console.print("\n[bold]Likely Capabilities:[/bold]")
            for cap in capabilities:
                console.print(f"  - {cap}")

        # Suggest related models
        console.print("\n[dim]Tip: Run 02_list_models.py to see all available models[/dim]")

    except NotFoundError:
        console.print(f"[red]Error:[/red] Model '{model_id}' not found.")
        console.print("\n[yellow]Available model suggestions:[/yellow]")
        console.print("  - grok-4-1-fast-reasoning (fast, efficient)")
        console.print("  - grok-3 (high intelligence)")
        console.print("  - grok-vision-beta (image understanding)")
        console.print("\nRun 02_list_models.py to see all available models.")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error fetching model info:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
