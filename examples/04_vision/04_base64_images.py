#!/usr/bin/env python3
"""
04_base64_images.py - Using Base64-Encoded Local Images

This example demonstrates how to send local images to Grok Vision using
base64 encoding. This is useful when you have images stored locally or
generated programmatically that you want to analyze.

Key concepts:
- Reading and encoding local images as base64
- Constructing data URIs for image content
- Handling different image formats (PNG, JPEG, GIF, WebP)
"""

import base64
import os
from io import BytesIO
from pathlib import Path

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


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """
    Read a local image file and encode it as base64.

    Args:
        image_path: Path to the local image file.

    Returns:
        Tuple of (base64_string, mime_type).
    """
    path = Path(image_path)

    # Determine MIME type based on file extension
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    extension = path.suffix.lower()
    mime_type = mime_types.get(extension, "image/png")

    with open(path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")

    return base64_string, mime_type


def analyze_base64_image(base64_string: str, mime_type: str, prompt: str) -> str:
    """
    Analyze a base64-encoded image.

    Args:
        base64_string: Base64-encoded image data.
        mime_type: MIME type of the image (e.g., 'image/png').
        prompt: Question or instruction about the image.

    Returns:
        The model's analysis of the image.
    """
    # Construct the data URI
    data_uri = f"data:{mime_type};base64,{base64_string}"

    response = client.chat.completions.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def create_sample_image() -> tuple[str, str]:
    """
    Create a simple sample image for demonstration.
    Uses PIL if available, otherwise creates a minimal valid PNG.

    Returns:
        Tuple of (base64_string, mime_type).
    """
    try:
        # Try to use PIL for a nicer sample image
        from PIL import Image, ImageDraw, ImageFont

        # Create a simple image with text
        img = Image.new("RGB", (400, 200), color=(73, 109, 137))
        draw = ImageDraw.Draw(img)

        # Draw some shapes and text
        draw.rectangle([20, 20, 380, 180], outline="white", width=3)
        draw.ellipse([50, 50, 150, 150], fill="yellow", outline="orange")
        draw.rectangle([250, 50, 350, 150], fill="red", outline="darkred")

        # Add text
        draw.text((160, 160), "Sample Image", fill="white")

        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return base64_string, "image/png"

    except ImportError:
        # Create a minimal valid 1x1 red PNG without PIL
        # This is a valid PNG file structure
        png_data = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,  # 1x1
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,
                0x54,  # IDAT chunk
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xCF,
                0xC0,
                0x00,
                0x00,
                0x00,
                0x03,
                0x00,
                0x01,
                0x00,
                0x18,
                0xDD,
                0x8D,
                0xB4,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,  # IEND chunk
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )
        base64_string = base64.b64encode(png_data).decode("utf-8")
        return base64_string, "image/png"


def analyze_local_image(image_path: str, prompt: str) -> str:
    """
    Convenience function to analyze a local image file.

    Args:
        image_path: Path to the local image file.
        prompt: Question or instruction about the image.

    Returns:
        The model's analysis of the image.
    """
    base64_string, mime_type = encode_image_to_base64(image_path)
    return analyze_base64_image(base64_string, mime_type, prompt)


def main():
    console.print(
        Panel.fit(
            "[bold blue]Base64 Image Analysis with Grok Vision[/bold blue]\n"
            "Analyze local images by encoding them as base64",
            border_style="blue",
        )
    )

    # Example 1: Create and analyze a programmatically generated image
    console.print("\n[bold yellow]Example 1: Programmatically Generated Image[/bold yellow]")

    base64_string, mime_type = create_sample_image()
    console.print(f"[dim]Created sample image ({mime_type})[/dim]")
    console.print(f"[dim]Base64 length: {len(base64_string)} characters[/dim]")

    analysis = analyze_base64_image(
        base64_string,
        mime_type,
        "Describe what you see in this image. What shapes and colors are present?",
    )
    console.print(Panel(analysis, title="Generated Image Analysis", border_style="green"))

    # Example 2: Demonstrate the structure of base64 image requests
    console.print("\n[bold yellow]Example 2: Base64 Data URI Structure[/bold yellow]")

    console.print(
        Panel(
            f"""[bold]Data URI Format:[/bold]
data:{mime_type};base64,<base64_encoded_data>

[bold]Components:[/bold]
- Scheme: data:
- MIME Type: {mime_type}
- Encoding: ;base64,
- Data: {base64_string[:50]}...

[bold]Full URI length:[/bold] {len(f"data:{mime_type};base64,{base64_string}")} characters""",
            title="Data URI Structure",
            border_style="cyan",
        )
    )

    # Example 3: Show how to handle a local file (if one exists)
    console.print("\n[bold yellow]Example 3: Local File Processing[/bold yellow]")

    # Look for any local image in common locations
    possible_paths = [
        "./sample_image.png",
        "./sample_image.jpg",
        "../sample_image.png",
        "~/Pictures/sample.png",
    ]

    local_image_found = False
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            console.print(f"[green]Found local image: {expanded_path}[/green]")
            local_image_found = True
            try:
                result = analyze_local_image(expanded_path, "Describe this image.")
                console.print(Panel(result, title="Local Image Analysis", border_style="magenta"))
            except Exception as e:
                console.print(f"[red]Error analyzing image: {e}[/red]")
            break

    if not local_image_found:
        console.print(
            Panel(
                """To analyze a local image, use the following code:

[bold]# Read and encode the image[/bold]
base64_string, mime_type = encode_image_to_base64("path/to/your/image.png")

[bold]# Analyze it[/bold]
result = analyze_base64_image(
    base64_string,
    mime_type,
    "Describe this image."
)

[bold]# Or use the convenience function[/bold]
result = analyze_local_image("path/to/image.png", "What's in this image?")""",
                title="Usage Instructions",
                border_style="yellow",
            )
        )

    # Show supported formats
    console.print("\n[bold yellow]Supported Image Formats:[/bold yellow]")
    console.print(
        Panel(
            """[bold]Supported MIME types:[/bold]
- image/jpeg (.jpg, .jpeg)
- image/png (.png)
- image/gif (.gif)
- image/webp (.webp)

[bold]Best practices:[/bold]
- Keep images under 20MB
- Use appropriate compression
- Prefer JPEG for photos, PNG for graphics
- Resize very large images before encoding""",
            border_style="dim",
        )
    )

    console.print(
        "\n[green]Base64 image processing complete![/green] "
        "Local images can be analyzed by encoding them as base64 data URIs."
    )


if __name__ == "__main__":
    main()
