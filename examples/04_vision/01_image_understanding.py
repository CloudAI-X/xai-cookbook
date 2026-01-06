#!/usr/bin/env python3
"""
01_image_understanding.py - Basic Image Understanding with Grok Vision

This example demonstrates how to use Grok's vision capabilities to analyze
and understand images. The model can describe image content, identify objects,
read text, and answer questions about visual content.

Vision-capable models include:
- grok-4-0709
- grok-4-fast-*
- grok-4-1-fast-*
- grok-2-vision-1212

Key concepts:
- Sending images via URL to vision models
- Multi-modal content (text + image) in messages
- Understanding image descriptions and analysis
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

# Sample images for demonstration
SAMPLE_IMAGES = {
    "nature": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-702-702.jpg/800px-24701-nature-702-702.jpg",
    "architecture": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/800px-Tour_Eiffel_Wikimedia_Commons.jpg",
}


def analyze_image(image_url: str, question: str = "What's in this image?") -> str:
    """
    Analyze an image and answer a question about it.

    Args:
        image_url: URL of the image to analyze.
        question: Question to ask about the image.

    Returns:
        The model's analysis of the image.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Image Understanding with Grok Vision[/bold blue]\n"
            "Analyze images using grok-4-1-fast-non-reasoning",
            border_style="blue",
        )
    )

    # Example 1: Basic image description
    console.print("\n[bold yellow]Example 1: Basic Image Description[/bold yellow]")
    console.print(f"[dim]Image URL: {SAMPLE_IMAGES['nature']}[/dim]")

    description = analyze_image(
        SAMPLE_IMAGES["nature"],
        "Describe this image in detail. What do you see?",
    )
    console.print(Panel(description, title="Image Description", border_style="green"))

    # Example 2: Specific question about an image
    console.print("\n[bold yellow]Example 2: Specific Question About Image[/bold yellow]")
    console.print(f"[dim]Image URL: {SAMPLE_IMAGES['architecture']}[/dim]")

    answer = analyze_image(
        SAMPLE_IMAGES["architecture"],
        "What famous landmark is this? What city is it located in?",
    )
    console.print(Panel(answer, title="Landmark Identification", border_style="cyan"))

    # Example 3: Detailed analysis with context
    console.print("\n[bold yellow]Example 3: Detailed Analysis[/bold yellow]")

    detailed_analysis = analyze_image(
        SAMPLE_IMAGES["architecture"],
        "Analyze this image from an architectural perspective. "
        "Describe the style, materials, and any notable features you can observe.",
    )
    console.print(Panel(detailed_analysis, title="Architectural Analysis", border_style="magenta"))

    console.print(
        "\n[green]Image understanding complete![/green] "
        "Grok can analyze various types of visual content."
    )


if __name__ == "__main__":
    main()
