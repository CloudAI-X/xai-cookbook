#!/usr/bin/env python3
"""
05_url_images.py - Using Images from URLs

This example demonstrates various ways to work with images from URLs,
including different image sources, URL validation, and handling various
image hosting scenarios.

Key concepts:
- Fetching images from public URLs
- Working with different image hosting services
- URL image best practices and error handling
"""

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()

# Collection of sample images from various sources
SAMPLE_IMAGES = {
    "wikimedia_nature": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-702-702.jpg/800px-24701-nature-702-702.jpg",
        "description": "Nature scene from Wikimedia Commons",
        "source": "Wikimedia Commons",
    },
    "wikimedia_architecture": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/800px-Tour_Eiffel_Wikimedia_Commons.jpg",
        "description": "Eiffel Tower from Wikimedia Commons",
        "source": "Wikimedia Commons",
    },
    "nasa_space": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/NGC_4414_%28NASA-med%29.jpg/800px-NGC_4414_%28NASA-med%29.jpg",
        "description": "NGC 4414 spiral galaxy",
        "source": "NASA via Wikimedia",
    },
}


def validate_image_url(url: str) -> dict:
    """
    Validate and analyze an image URL.

    Args:
        url: The URL to validate.

    Returns:
        Dictionary with validation results.
    """
    result = {
        "url": url,
        "is_valid": True,
        "scheme": "",
        "domain": "",
        "path": "",
        "warnings": [],
    }

    try:
        parsed = urlparse(url)
        result["scheme"] = parsed.scheme
        result["domain"] = parsed.netloc
        result["path"] = parsed.path

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            result["is_valid"] = False
            result["warnings"].append("URL must use http or https scheme")

        # Check for common issues
        if not parsed.netloc:
            result["is_valid"] = False
            result["warnings"].append("URL missing domain")

        if parsed.scheme == "http":
            result["warnings"].append("Consider using HTTPS for security")

        # Check file extension (basic check)
        valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        path_lower = parsed.path.lower()
        has_valid_ext = any(path_lower.endswith(ext) for ext in valid_extensions)

        if not has_valid_ext and "." in path_lower.split("/")[-1]:
            result["warnings"].append("Unusual file extension - may not be an image")

    except Exception as e:
        result["is_valid"] = False
        result["warnings"].append(f"URL parsing error: {str(e)}")

    return result


def analyze_url_image(image_url: str, prompt: str) -> str:
    """
    Analyze an image from a URL.

    Args:
        image_url: Public URL of the image.
        prompt: Question or instruction about the image.

    Returns:
        The model's analysis of the image.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def analyze_with_detail_level(image_url: str, prompt: str, detail: str = "auto") -> str:
    """
    Analyze an image with specified detail level.

    The detail parameter can be:
    - "auto": Let the model decide (default)
    - "low": Faster processing, lower detail
    - "high": Higher detail analysis, more tokens

    Args:
        image_url: Public URL of the image.
        prompt: Question or instruction about the image.
        detail: Detail level for image processing.

    Returns:
        The model's analysis of the image.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": detail},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]URL Image Analysis with Grok Vision[/bold blue]\n"
            "Analyze images directly from public URLs",
            border_style="blue",
        )
    )

    # Example 1: Show available sample images
    console.print("\n[bold yellow]Example 1: Available Sample Images[/bold yellow]")

    table = Table(title="Sample Images")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Description", style="white")

    for name, info in SAMPLE_IMAGES.items():
        table.add_row(name, info["source"], info["description"])

    console.print(table)

    # Example 2: URL validation
    console.print("\n[bold yellow]Example 2: URL Validation[/bold yellow]")

    test_urls = [
        SAMPLE_IMAGES["wikimedia_nature"]["url"],
        "https://example.com/image.png",
        "http://insecure-site.com/photo.jpg",
        "invalid-url",
    ]

    for url in test_urls:
        validation = validate_image_url(url)
        status = "[green]Valid[/green]" if validation["is_valid"] else "[red]Invalid[/red]"
        console.print(
            f"\n{status}: [dim]{url[:60]}...[/dim]"
            if len(url) > 60
            else f"\n{status}: [dim]{url}[/dim]"
        )

        if validation["warnings"]:
            for warning in validation["warnings"]:
                console.print(f"  [yellow]Warning:[/yellow] {warning}")

    # Example 3: Analyze images from different sources
    console.print("\n[bold yellow]Example 3: Analyze Images from Different Sources[/bold yellow]")

    for name, info in SAMPLE_IMAGES.items():
        console.print(f"\n[bold cyan]Analyzing: {info['description']}[/bold cyan]")
        console.print(f"[dim]Source: {info['source']}[/dim]")

        try:
            analysis = analyze_url_image(
                info["url"],
                "Briefly describe what you see in this image in 2-3 sentences.",
            )
            console.print(Panel(analysis, border_style="green"))
        except Exception as e:
            console.print(f"[red]Error analyzing image: {e}[/red]")

    # Example 4: Detail level comparison
    console.print("\n[bold yellow]Example 4: Detail Level Options[/bold yellow]")

    sample_url = SAMPLE_IMAGES["wikimedia_architecture"]["url"]

    console.print(
        Panel(
            """[bold]Detail levels available:[/bold]

[cyan]auto[/cyan] (default): Model decides the appropriate detail level
[cyan]low[/cyan]: Faster processing, uses fewer tokens, good for simple analysis
[cyan]high[/cyan]: Maximum detail, uses more tokens, good for detailed analysis

[bold]Usage:[/bold]
{"type": "image_url", "image_url": {"url": "...", "detail": "high"}}""",
            title="Image Detail Levels",
            border_style="cyan",
        )
    )

    console.print("\n[dim]Analyzing with different detail levels...[/dim]")

    # Low detail
    console.print("\n[bold]Low Detail Analysis:[/bold]")
    low_detail = analyze_with_detail_level(
        sample_url,
        "What is the main subject of this image?",
        detail="low",
    )
    console.print(Panel(low_detail, border_style="yellow"))

    # High detail
    console.print("\n[bold]High Detail Analysis:[/bold]")
    high_detail = analyze_with_detail_level(
        sample_url,
        "Describe this image in detail, including any text, small details, or background elements.",
        detail="high",
    )
    console.print(Panel(high_detail, border_style="magenta"))

    # Best practices
    console.print("\n[bold yellow]URL Image Best Practices:[/bold yellow]")
    console.print(
        Panel(
            """[bold]Recommendations for URL images:[/bold]

1. [cyan]Use HTTPS[/cyan]: Always prefer HTTPS URLs for security
2. [cyan]Public access[/cyan]: Ensure images are publicly accessible (no auth required)
3. [cyan]Stable URLs[/cyan]: Use permanent URLs, avoid temporary links
4. [cyan]Reasonable size[/cyan]: Keep images under 20MB
5. [cyan]Standard formats[/cyan]: Use JPEG, PNG, GIF, or WebP
6. [cyan]Direct links[/cyan]: Use direct image URLs, not webpage URLs

[bold]Reliable image sources:[/bold]
- Wikimedia Commons
- NASA Image Gallery
- Unsplash (direct image links)
- Your own CDN or cloud storage (S3, GCS, etc.)""",
            border_style="dim",
        )
    )

    console.print(
        "\n[green]URL image analysis complete![/green] "
        "Grok Vision can analyze images from any publicly accessible URL."
    )


if __name__ == "__main__":
    main()
