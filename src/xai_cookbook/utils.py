"""Shared utilities for xAI cookbook examples."""

import base64
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def print_response(content: str, title: str = "Response") -> None:
    """Print a formatted response panel."""
    console.print(Panel(Markdown(content), title=title, border_style="green"))


def print_code(code: str, language: str = "python", title: str = "Code") -> None:
    """Print syntax-highlighted code."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=title, border_style="blue"))


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red bold]Error:[/red bold] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[cyan]ℹ[/cyan] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def encode_image_to_base64(image_path: str | Path) -> str:
    """Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    path = Path(image_path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str | Path) -> str:
    """Get the media type for an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Media type string (e.g., 'image/png', 'image/jpeg')
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "image/png")


def create_image_content(image_path: str | Path) -> dict[str, object]:
    """Create an image content block for the API.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with image_url content block
    """
    base64_image = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
    }


def create_url_image_content(url: str) -> dict[str, object]:
    """Create an image content block from a URL.

    Args:
        url: URL of the image

    Returns:
        Dictionary with image_url content block
    """
    return {"type": "image_url", "image_url": {"url": url}}
