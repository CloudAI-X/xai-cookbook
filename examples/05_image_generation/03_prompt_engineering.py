#!/usr/bin/env python3
"""
03_prompt_engineering.py - Effective Prompts for Better Images

This script demonstrates techniques for crafting effective prompts to get
higher quality and more accurate image generations. Learn how to structure
your prompts for optimal results.

Key concepts:
- Structuring prompts with subject, style, and details
- Using descriptive adjectives and modifiers
- Specifying artistic styles and mediums
- Adding lighting, mood, and composition details
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


def generate_image(prompt: str) -> str:
    """Generate a single image from a text prompt."""
    client = get_client()
    response = client.images.generate(
        model="grok-2-image-1212",
        prompt=prompt,
        n=1,
    )
    return response.data[0].url


def build_structured_prompt(
    subject: str,
    style: str = "",
    lighting: str = "",
    mood: str = "",
    composition: str = "",
    details: str = "",
) -> str:
    """
    Build a well-structured prompt from components.

    Args:
        subject: The main subject of the image.
        style: Artistic style (e.g., "oil painting", "digital art").
        lighting: Lighting description (e.g., "golden hour", "dramatic").
        mood: Emotional tone (e.g., "peaceful", "mysterious").
        composition: Camera/view description (e.g., "close-up", "wide angle").
        details: Additional specific details.

    Returns:
        A well-formatted prompt string.
    """
    parts = [subject]

    if style:
        parts.append(f"in {style} style")
    if lighting:
        parts.append(f"with {lighting} lighting")
    if mood:
        parts.append(f"{mood} mood")
    if composition:
        parts.append(f"{composition} shot")
    if details:
        parts.append(details)

    return ", ".join(parts)


def main():
    console.print(
        Panel.fit(
            "[bold blue]Prompt Engineering for Image Generation[/bold blue]\n"
            "Learn techniques for crafting effective prompts",
            border_style="blue",
        )
    )

    # Technique 1: Basic vs Enhanced Prompts
    console.print("\n[bold yellow]Technique 1: Basic vs Enhanced Prompts[/bold yellow]")

    basic_prompt = "a cat"
    enhanced_prompt = (
        "A fluffy orange tabby cat lounging on a velvet cushion, "
        "soft natural window light, photorealistic, shallow depth of field"
    )

    comparison_table = Table(title="Prompt Comparison", show_lines=True)
    comparison_table.add_column("Type", style="yellow")
    comparison_table.add_column("Prompt", style="green", max_width=50)
    comparison_table.add_row("Basic", basic_prompt)
    comparison_table.add_row("Enhanced", enhanced_prompt)
    console.print(comparison_table)

    console.print("\n[dim]Generating enhanced version...[/dim]")
    url = generate_image(enhanced_prompt)
    console.print(f"[bold cyan]Enhanced Result:[/bold cyan]\n{url}")

    # Technique 2: Using the Structured Prompt Builder
    console.print("\n[bold yellow]Technique 2: Structured Prompt Building[/bold yellow]")

    structured_prompt = build_structured_prompt(
        subject="A wise old wizard reading an ancient spellbook",
        style="fantasy digital art",
        lighting="candlelight",
        mood="mysterious and magical",
        composition="portrait",
        details="intricate robes, flowing white beard, glowing runes",
    )

    console.print(f"[bold green]Built Prompt:[/bold green]\n{structured_prompt}\n")

    console.print("[dim]Generating image...[/dim]")
    url = generate_image(structured_prompt)
    console.print(f"[bold cyan]Result:[/bold cyan]\n{url}")

    # Technique 3: Style Modifiers
    console.print("\n[bold yellow]Technique 3: Style Modifiers[/bold yellow]")

    style_modifiers = {
        "Photorealistic": "photorealistic, 8k resolution, highly detailed",
        "Oil Painting": "oil painting style, visible brushstrokes, rich colors",
        "Anime": "anime style, vibrant colors, cel shading",
        "Watercolor": "watercolor painting, soft edges, flowing colors",
        "Cyberpunk": "cyberpunk aesthetic, neon lights, futuristic",
    }

    console.print("[bold green]Common Style Modifiers:[/bold green]")
    for name, modifier in style_modifiers.items():
        console.print(f"  [cyan]{name}:[/cyan] {modifier}")

    # Generate one example with style modifier
    subject = "A bustling marketplace"
    chosen_style = "cyberpunk aesthetic, neon lights, rain-slicked streets"
    styled_prompt = f"{subject}, {chosen_style}"

    console.print(f"\n[bold green]Example Prompt:[/bold green] {styled_prompt}")
    console.print("[dim]Generating image...[/dim]")
    url = generate_image(styled_prompt)
    console.print(f"[bold cyan]Result:[/bold cyan]\n{url}")

    # Technique 4: Composition and Camera Terms
    console.print("\n[bold yellow]Technique 4: Composition Terms[/bold yellow]")

    composition_terms = [
        "close-up shot - focus on details",
        "wide angle - expansive scenes",
        "bird's eye view - top-down perspective",
        "low angle - dramatic, powerful subjects",
        "rule of thirds - balanced composition",
        "bokeh background - blurred background, sharp subject",
    ]

    console.print("[bold green]Useful Composition Terms:[/bold green]")
    for term in composition_terms:
        console.print(f"  [cyan]-[/cyan] {term}")

    # Example with composition
    composition_prompt = (
        "A lone astronaut standing on Mars, "
        "wide angle shot, dramatic low angle, "
        "Earth visible in the sky, cinematic lighting"
    )
    console.print(f"\n[bold green]Example Prompt:[/bold green] {composition_prompt}")
    console.print("[dim]Generating image...[/dim]")
    url = generate_image(composition_prompt)
    console.print(f"[bold cyan]Result:[/bold cyan]\n{url}")

    # Technique 5: Negative Guidance (what to avoid)
    console.print("\n[bold yellow]Technique 5: Being Specific[/bold yellow]")

    tips = [
        "Be specific about colors: 'deep crimson' vs 'red'",
        "Specify materials: 'weathered oak wood' vs 'wood'",
        "Include environment: 'in a misty forest' vs no context",
        "Add time of day: 'at golden hour' for warm lighting",
        "Mention quality: 'highly detailed', 'professional quality'",
    ]

    console.print("[bold green]Pro Tips for Better Results:[/bold green]")
    for tip in tips:
        console.print(f"  [cyan]-[/cyan] {tip}")

    # Final example combining all techniques
    console.print("\n[bold yellow]Final Example: Combining All Techniques[/bold yellow]")

    master_prompt = (
        "A majestic phoenix rising from golden flames, "
        "fantasy digital art style, "
        "dramatic backlighting with warm orange and red tones, "
        "epic and triumphant mood, "
        "dynamic low angle composition, "
        "intricate feather details with glowing embers, "
        "dark stormy sky background, "
        "highly detailed, cinematic quality"
    )

    console.print(f"[bold green]Master Prompt:[/bold green]\n{master_prompt}\n")
    console.print("[dim]Generating image...[/dim]")
    url = generate_image(master_prompt)
    console.print(f"[bold cyan]Result:[/bold cyan]\n{url}")

    console.print(
        Panel(
            "[green]Prompt engineering examples complete![/green]\n\n"
            "Remember:\n"
            "- Be specific and descriptive\n"
            "- Include style, lighting, mood, and composition\n"
            "- Use quality modifiers for better results\n"
            "- Experiment and iterate on your prompts",
            title="Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
