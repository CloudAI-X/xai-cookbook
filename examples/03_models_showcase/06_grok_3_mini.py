#!/usr/bin/env python3
"""
Grok-3 Mini - Cost-Effective Performance

Model: grok-3-mini
Context Window: 131K tokens
Pricing: $0.30/1M input, $0.50/1M output (10x cheaper!)

Grok-3 Mini is optimized for:
- Cost-sensitive applications
- High-volume processing
- Quick responses
- Standard tasks that don't need flagship capability
- Prototyping and development

Best for: Budget-conscious applications with good performance needs
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


def create_client() -> OpenAI:
    """Create xAI client using OpenAI SDK."""
    return OpenAI(api_key=os.environ["X_AI_API_KEY"], base_url="https://api.x.ai/v1")


def demonstrate_cost_efficiency(client: OpenAI) -> None:
    """Show cost comparison and efficiency."""
    console.print(
        Panel.fit(
            "[bold cyan]Grok-3 Mini Demo[/bold cyan]\n"
            "Model: grok-3-mini | Context: 131K | $0.30/$0.50 per 1M tokens",
            border_style="cyan",
        )
    )

    # Cost comparison table
    table = Table(title="Cost Comparison (per 1M tokens)")
    table.add_column("Model", style="cyan")
    table.add_column("Input Cost", justify="right")
    table.add_column("Output Cost", justify="right")
    table.add_column("Savings vs Grok-4", justify="right", style="green")

    table.add_row("grok-4-0709", "$3.00", "$15.00", "-")
    table.add_row("grok-3", "$3.00", "$15.00", "-")
    table.add_row(
        "[bold]grok-3-mini[/bold]",
        "[bold]$0.30[/bold]",
        "[bold]$0.50[/bold]",
        "[bold]90%+[/bold]",
    )

    console.print("\n")
    console.print(table)
    console.print("\n[yellow]Grok-3 Mini offers excellent value for many use cases![/yellow]\n")


def demonstrate_classification(client: OpenAI) -> None:
    """Demonstrate text classification - a common high-volume task."""
    console.print("=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Text Classification (High-Volume Use Case)\n")

    texts = [
        "I absolutely love this product! Best purchase ever!",
        "The item arrived damaged and customer service was unhelpful.",
        "It's okay, nothing special. Does what it's supposed to do.",
        "Shipping was fast but the quality could be better.",
        "Would definitely recommend to friends and family!",
    ]

    console.print("[bold]Classifying customer reviews:[/bold]\n")

    for text in texts:
        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Classify the sentiment as POSITIVE, NEGATIVE, or NEUTRAL. "
                    "Respond with only the classification.",
                },
                {"role": "user", "content": f"Review: {text}"},
            ],
            temperature=0.1,
            max_tokens=10,
        )

        sentiment = response.choices[0].message.content.strip()
        color = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "yellow"}.get(
            sentiment, "white"
        )
        console.print(f"[dim]{text[:50]}...[/dim]")
        console.print(f"  -> [{color}]{sentiment}[/{color}]\n")


def demonstrate_entity_extraction(client: OpenAI) -> None:
    """Demonstrate entity extraction at scale."""
    console.print("=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Entity Extraction\n")

    text = """
    Meeting scheduled with Dr. Sarah Johnson at Microsoft headquarters in Seattle
    on March 15, 2025. Contact her at sarah.johnson@microsoft.com or call
    (425) 555-0123. Budget approved: $50,000 for the Q2 project.
    """

    console.print(Panel(text.strip(), title="Input Text", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-3-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract entities from text. Return as a simple list with categories.",
            },
            {
                "role": "user",
                "content": f"Extract all named entities (people, organizations, locations, "
                f"dates, emails, phone numbers, money amounts):\n\n{text}",
            },
        ],
        temperature=0.1,
        max_tokens=300,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Extracted Entities:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_translation(client: OpenAI) -> None:
    """Demonstrate translation capabilities."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Quick Translation\n")

    text = "The quick brown fox jumps over the lazy dog."
    languages = ["Spanish", "French", "German", "Japanese"]

    console.print(f"[yellow]Original:[/yellow] {text}\n")

    for lang in languages:
        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Translate to {lang}. Return only the translation.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=100,
        )

        translation = response.choices[0].message.content
        console.print(f"[cyan]{lang}:[/cyan] {translation}")


def demonstrate_formatting(client: OpenAI) -> None:
    """Demonstrate text formatting and cleanup."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Text Formatting\n")

    messy_text = """
    john smith,ceo,acme corp,john@acme.com,555-1234
    jane doe,cto,techstart,jane@techstart.io,555-5678
    bob wilson,vp sales,globalco,bob@globalco.com,555-9012
    """

    console.print(Panel(messy_text.strip(), title="Raw CSV Data", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-3-mini",
        messages=[
            {
                "role": "system",
                "content": "Format data into clean, readable tables or lists.",
            },
            {
                "role": "user",
                "content": f"Format this CSV data into a nice markdown table:\n{messy_text}",
            },
        ],
        temperature=0.2,
        max_tokens=300,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Formatted Output:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Grok-3 Mini Model Demonstration[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    demonstrate_cost_efficiency(client)
    demonstrate_classification(client)
    demonstrate_entity_extraction(client)
    demonstrate_translation(client)
    demonstrate_formatting(client)

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print("[dim]Grok-3 Mini: Great performance at 90% lower cost![/dim]\n")
