#!/usr/bin/env python3
"""
02_multiple_images.py - Process Multiple Images in One Request

This example demonstrates how to send multiple images in a single API request
for comparison, analysis, or batch processing. Grok can analyze relationships
between images, compare visual content, and provide unified analysis.

Key concepts:
- Sending multiple images in one message
- Comparing and contrasting visual content
- Batch image analysis for efficiency
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()

# Sample images for comparison
COMPARISON_IMAGES = {
    "sunset": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/800px-GoldenGateBridge-001.jpg",
    "bridge": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Sydney_Harbour_Bridge_from_Circular_Quay.jpg/800px-Sydney_Harbour_Bridge_from_Circular_Quay.jpg",
}


def analyze_multiple_images(image_urls: list[str], prompt: str) -> str:
    """
    Analyze multiple images together in a single request.

    Args:
        image_urls: List of image URLs to analyze.
        prompt: Question or instruction about the images.

    Returns:
        The model's analysis of all images.
    """
    # Build content list with text prompt and all images
    content = [{"type": "text", "text": prompt}]

    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Multiple Image Analysis with Grok Vision[/bold blue]\n"
            "Compare and analyze multiple images in a single request",
            border_style="blue",
        )
    )

    # Get the image URLs as a list
    image_urls = list(COMPARISON_IMAGES.values())

    console.print("\n[bold yellow]Images being analyzed:[/bold yellow]")
    for name, url in COMPARISON_IMAGES.items():
        console.print(f"  [dim]{name}: {url}[/dim]")

    # Example 1: Compare images
    console.print("\n[bold yellow]Example 1: Image Comparison[/bold yellow]")

    comparison = analyze_multiple_images(
        image_urls,
        "Compare these two images. What are the similarities and differences? "
        "Focus on the bridges and their architectural styles.",
    )
    console.print(Panel(comparison, title="Image Comparison", border_style="green"))

    # Example 2: Find common themes
    console.print("\n[bold yellow]Example 2: Common Themes[/bold yellow]")

    themes = analyze_multiple_images(
        image_urls,
        "What common themes or elements do these images share? "
        "Are there any visual patterns or subjects that appear in both?",
    )
    console.print(Panel(themes, title="Common Themes", border_style="cyan"))

    # Example 3: Preference analysis
    console.print("\n[bold yellow]Example 3: Photography Analysis[/bold yellow]")

    photography = analyze_multiple_images(
        image_urls,
        "Analyze these images from a photography perspective. "
        "Compare the composition, lighting, and visual impact of each.",
    )
    console.print(Panel(photography, title="Photography Analysis", border_style="magenta"))

    # Example 4: Batch description
    console.print("\n[bold yellow]Example 4: Batch Descriptions[/bold yellow]")

    batch = analyze_multiple_images(
        image_urls,
        "Provide a brief one-sentence description for each image, numbered 1 and 2.",
    )
    console.print(Panel(batch, title="Batch Descriptions", border_style="yellow"))

    console.print(
        "\n[green]Multi-image analysis complete![/green] "
        "Grok can efficiently process and compare multiple images."
    )


if __name__ == "__main__":
    main()
