#!/usr/bin/env python3
"""
02_multiple_images.py - Generate Multiple Images in Batch

This script demonstrates how to generate multiple images from a single prompt
or process multiple prompts in sequence. This is useful for generating
variations or creating a collection of related images.

Key concepts:
- Using n parameter to generate multiple images at once
- Processing multiple prompts sequentially
- Organizing and displaying batch results
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


def get_client() -> OpenAI:
    """Get an authenticated OpenAI client for xAI."""
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] X_AI_API_KEY environment variable not set.\n"
            "Please set it in your .env file or environment."
        )
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def generate_multiple_images(prompt: str, count: int = 2) -> list[str]:
    """
    Generate multiple images from a single prompt.

    Args:
        prompt: Text description of the image to generate.
        count: Number of images to generate (default: 2).

    Returns:
        List of URLs for the generated images.
    """
    client = get_client()

    response = client.images.generate(
        model="grok-2-image-1212",
        prompt=prompt,
        n=count,
    )

    return [image.url for image in response.data]


def generate_batch_prompts(prompts: list[str]) -> dict[str, str]:
    """
    Generate one image for each prompt in a list.

    Args:
        prompts: List of text prompts.

    Returns:
        Dictionary mapping prompts to their generated image URLs.
    """
    client = get_client()
    results = {}

    for prompt in prompts:
        console.print(f"[dim]Generating: {prompt[:50]}...[/dim]")
        response = client.images.generate(
            model="grok-2-image-1212",
            prompt=prompt,
            n=1,
        )
        results[prompt] = response.data[0].url

    return results


def main():
    console.print(
        Panel.fit(
            "[bold blue]Multiple Image Generation[/bold blue]\n"
            "Generate multiple images in batch using grok-2-image-1212",
            border_style="blue",
        )
    )

    # Example 1: Multiple variations from single prompt
    console.print("\n[bold yellow]Example 1: Multiple Variations (n=2)[/bold yellow]")
    prompt = "A futuristic city skyline at sunset with flying vehicles"
    console.print(f"[bold green]Prompt:[/bold green] {prompt}\n")

    console.print("[dim]Generating 2 image variations...[/dim]")
    urls = generate_multiple_images(prompt, count=2)

    for i, url in enumerate(urls, 1):
        console.print(f"\n[bold cyan]Variation {i}:[/bold cyan]")
        console.print(url)

    # Example 2: Batch of different prompts
    console.print("\n[bold yellow]Example 2: Batch of Different Prompts[/bold yellow]")

    prompts = [
        "A serene Japanese garden with cherry blossoms",
        "A medieval castle on a misty mountain",
        "An underwater coral reef teeming with colorful fish",
    ]

    console.print("[bold green]Prompts to process:[/bold green]")
    for i, p in enumerate(prompts, 1):
        console.print(f"  {i}. {p}")
    console.print()

    results = generate_batch_prompts(prompts)

    # Display results in a table
    table = Table(title="Batch Generation Results", show_lines=True)
    table.add_column("Prompt", style="green", max_width=40)
    table.add_column("Generated URL", style="cyan")

    for prompt, url in results.items():
        # Truncate prompt for display
        display_prompt = prompt[:37] + "..." if len(prompt) > 40 else prompt
        table.add_row(display_prompt, url)

    console.print("\n")
    console.print(table)

    # Example 3: Theme variations
    console.print("\n[bold yellow]Example 3: Theme Variations[/bold yellow]")

    base_subject = "a majestic lion"
    styles = ["photorealistic", "watercolor painting", "digital art"]

    themed_prompts = [f"{base_subject} in {style} style" for style in styles]

    console.print(f"[bold green]Base subject:[/bold green] {base_subject}")
    console.print(f"[bold green]Styles:[/bold green] {', '.join(styles)}\n")

    themed_results = generate_batch_prompts(themed_prompts)

    for style, (prompt, url) in zip(styles, themed_results.items()):
        console.print(f"\n[bold magenta]{style.title()}:[/bold magenta]")
        console.print(url)

    console.print(
        Panel(
            "[green]Batch generation complete![/green]\n\n"
            "You can use n>1 for variations of the same prompt,\n"
            "or process multiple prompts sequentially for different images.",
            title="Complete",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
