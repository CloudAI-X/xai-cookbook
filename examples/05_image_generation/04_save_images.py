#!/usr/bin/env python3
"""
04_save_images.py - Save Generated Images to Disk

This script demonstrates how to download and save generated images to your
local filesystem. Learn how to fetch images from URLs, save them with
meaningful names, and organize your generated image collection.

Key concepts:
- Downloading images from generated URLs
- Saving images with timestamps and descriptive names
- Creating organized output directories
- Handling different image formats
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

console = Console()

# Default output directory for saved images
OUTPUT_DIR = Path("generated_images")


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


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Convert text to a safe filename.

    Args:
        text: The text to convert.
        max_length: Maximum length of the resulting filename.

    Returns:
        A sanitized filename string.
    """
    # Remove or replace invalid characters
    safe_text = re.sub(r'[<>:"/\\|?*]', "", text)
    # Replace spaces and multiple underscores
    safe_text = re.sub(r"\s+", "_", safe_text)
    safe_text = re.sub(r"_+", "_", safe_text)
    # Truncate and remove trailing underscores
    return safe_text[:max_length].strip("_").lower()


def download_image(url: str, filepath: Path) -> bool:
    """
    Download an image from a URL and save it to disk.

    Args:
        url: The URL of the image to download.
        filepath: The path where the image should be saved.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()

            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write the image data
            with open(filepath, "wb") as f:
                f.write(response.content)

            return True

    except httpx.HTTPError as e:
        console.print(f"[red]HTTP Error downloading image:[/red] {e}")
        return False
    except OSError as e:
        console.print(f"[red]Error saving file:[/red] {e}")
        return False


def get_file_extension(url: str, default: str = ".png") -> str:
    """
    Extract file extension from URL or return default.

    Args:
        url: The image URL.
        default: Default extension if none found.

    Returns:
        File extension including the dot.
    """
    parsed = urlparse(url)
    path = parsed.path.lower()

    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        if path.endswith(ext):
            return ext

    return default


def generate_and_save(
    prompt: str,
    output_dir: Path = OUTPUT_DIR,
    filename_prefix: str = "",
) -> Path | None:
    """
    Generate an image and save it to disk.

    Args:
        prompt: The text prompt for image generation.
        output_dir: Directory to save images to.
        filename_prefix: Optional prefix for the filename.

    Returns:
        Path to the saved image, or None if failed.
    """
    # Generate the image
    console.print(f"[dim]Generating image for: {prompt[:50]}...[/dim]")
    url = generate_image(prompt)

    # Create filename from prompt and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = sanitize_filename(prompt)
    extension = get_file_extension(url)

    if filename_prefix:
        filename = f"{filename_prefix}_{safe_prompt}_{timestamp}{extension}"
    else:
        filename = f"{safe_prompt}_{timestamp}{extension}"

    filepath = output_dir / filename

    # Download and save
    console.print(f"[dim]Downloading to: {filepath}[/dim]")
    if download_image(url, filepath):
        return filepath
    return None


def main():
    console.print(
        Panel.fit(
            "[bold blue]Save Generated Images to Disk[/bold blue]\n"
            "Download and organize your generated images locally",
            border_style="blue",
        )
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[bold green]Output directory:[/bold green] {OUTPUT_DIR.absolute()}")

    # Example 1: Generate and save a single image
    console.print("\n[bold yellow]Example 1: Single Image Generation and Save[/bold yellow]")

    prompt1 = "A peaceful mountain lake at sunrise with perfect reflections"
    filepath = generate_and_save(prompt1)

    if filepath:
        console.print(f"[bold cyan]Saved to:[/bold cyan] {filepath}")
        console.print(f"[dim]File size: {filepath.stat().st_size / 1024:.1f} KB[/dim]")

    # Example 2: Generate multiple images with prefixes
    console.print("\n[bold yellow]Example 2: Batch Generation with Prefixes[/bold yellow]")

    prompts = [
        ("nature", "A dense rainforest with sunlight filtering through the canopy"),
        ("urban", "A quiet city alley with vintage neon signs at night"),
        ("abstract", "Flowing geometric patterns in blue and gold"),
    ]

    saved_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating images...", total=len(prompts))

        for prefix, prompt in prompts:
            filepath = generate_and_save(prompt, filename_prefix=prefix)
            if filepath:
                saved_files.append(filepath)
            progress.advance(task)

    console.print("\n[bold green]Saved files:[/bold green]")
    for fp in saved_files:
        console.print(f"  [cyan]-[/cyan] {fp.name}")

    # Example 3: Organized subdirectories
    console.print("\n[bold yellow]Example 3: Organized Subdirectories[/bold yellow]")

    categories = {
        "landscapes": "A dramatic desert landscape with towering red rock formations",
        "portraits": "A wise elderly man with kind eyes, cinematic portrait lighting",
        "fantasy": "A magical floating island with waterfalls cascading into clouds",
    }

    for category, prompt in categories.items():
        subdir = OUTPUT_DIR / category
        filepath = generate_and_save(prompt, output_dir=subdir)
        if filepath:
            console.print(f"  [cyan]{category}/[/cyan] {filepath.name}")

    # Summary
    total_files = list(OUTPUT_DIR.rglob("*.png")) + list(OUTPUT_DIR.rglob("*.jpg"))
    total_size = sum(f.stat().st_size for f in total_files)

    console.print(
        Panel(
            f"[green]Image saving complete![/green]\n\n"
            f"Output directory: {OUTPUT_DIR.absolute()}\n"
            f"Total images saved: {len(total_files)}\n"
            f"Total size: {total_size / 1024:.1f} KB\n\n"
            "Tips:\n"
            "- Use meaningful prefixes to organize images\n"
            "- Create subdirectories for different categories\n"
            "- Include timestamps to avoid overwriting",
            title="Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
