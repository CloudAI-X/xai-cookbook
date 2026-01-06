#!/usr/bin/env python3
"""
01_basic_generation.py - Simple Text-to-Image Generation

This script demonstrates the simplest form of image generation using the xAI API.
It sends a text prompt to grok-2-image-1212 and receives a generated image URL.

Key concepts:
- Using the images.generate() endpoint
- Crafting descriptive prompts
- Accessing generated image URLs
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()


def generate_image(prompt: str) -> str:
    """
    Generate a single image from a text prompt.

    Args:
        prompt: Text description of the image to generate.

    Returns:
        URL of the generated image.
    """
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] X_AI_API_KEY environment variable not set.\n"
            "Please set it in your .env file or environment."
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    response = client.images.generate(
        model="grok-2-image-1212",
        prompt=prompt,
        n=1,
    )

    return response.data[0].url


def main():
    console.print(
        Panel.fit(
            "[bold blue]Basic Image Generation[/bold blue]\n"
            "Generate images from text descriptions using grok-2-image-1212",
            border_style="blue",
        )
    )

    # Example 1: Simple descriptive prompt
    console.print("\n[bold yellow]Example 1: Simple Descriptive Prompt[/bold yellow]")
    prompt1 = "A golden retriever puppy playing in autumn leaves"
    console.print(f"[bold green]Prompt:[/bold green] {prompt1}\n")

    console.print("[dim]Generating image...[/dim]")
    url1 = generate_image(prompt1)
    console.print(f"[bold cyan]Generated Image URL:[/bold cyan]\n{url1}")

    # Example 2: More detailed prompt
    console.print("\n[bold yellow]Example 2: Detailed Scene Description[/bold yellow]")
    prompt2 = (
        "A cozy coffee shop interior with warm lighting, "
        "wooden tables, and plants by the window on a rainy day"
    )
    console.print(f"[bold green]Prompt:[/bold green] {prompt2}\n")

    console.print("[dim]Generating image...[/dim]")
    url2 = generate_image(prompt2)
    console.print(f"[bold cyan]Generated Image URL:[/bold cyan]\n{url2}")

    # Example 3: Abstract concept
    console.print("\n[bold yellow]Example 3: Abstract Concept[/bold yellow]")
    prompt3 = "The feeling of hope visualized as colorful light breaking through clouds"
    console.print(f"[bold green]Prompt:[/bold green] {prompt3}\n")

    console.print("[dim]Generating image...[/dim]")
    url3 = generate_image(prompt3)
    console.print(f"[bold cyan]Generated Image URL:[/bold cyan]\n{url3}")

    console.print(
        Panel(
            "[green]All images generated successfully![/green]\n\n"
            "Copy and paste the URLs above into your browser to view the images.",
            title="Complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
