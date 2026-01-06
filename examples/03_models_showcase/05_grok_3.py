#!/usr/bin/env python3
"""
Grok-3 - Powerful General-Purpose Model

Model: grok-3
Context Window: 131K tokens
Pricing: $3/1M input, $15/1M output

Grok-3 is a highly capable model offering:
- Strong general-purpose performance
- Good balance of capability and speed
- Solid reasoning abilities
- Quality text generation
- Reliable instruction following

Best for: General tasks requiring strong performance
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

console = Console()


def create_client() -> OpenAI:
    """Create xAI client using OpenAI SDK."""
    return OpenAI(api_key=os.environ["X_AI_API_KEY"], base_url="https://api.x.ai/v1")


def demonstrate_general_capability(client: OpenAI) -> None:
    """Demonstrate Grok-3's general-purpose capabilities."""
    console.print(
        Panel.fit(
            "[bold cyan]Grok-3 Model Demo[/bold cyan]\n"
            "Model: grok-3 | Context: 131K | $3/$15 per 1M tokens",
            border_style="cyan",
        )
    )

    # Writing task
    prompt = """Write a professional email to a client explaining a project delay.

Context:
- Original deadline: January 15th
- New deadline: January 29th
- Reason: Unexpected technical challenges with third-party API integration
- Impact: No change to final quality or cost
- Mitigation: Weekly progress updates offered

Keep it professional, apologetic but confident, and around 150-200 words."""

    console.print("\n[bold]Task:[/bold] Professional Email Writing\n")
    console.print(Panel(prompt, title="Request", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": "You are a professional business communications expert. "
                "Write clear, polished, and appropriately toned messages.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-3 Response:[/bold green]\n")
    console.print(Panel(content, border_style="green"))

    if response.usage:
        console.print(
            f"\n[dim]Tokens used - Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}[/dim]"
        )


def demonstrate_analysis(client: OpenAI) -> None:
    """Demonstrate analytical capabilities."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Data Analysis\n")

    prompt = """Analyze this sales data and provide insights:

Q1 2024 Sales by Region:
- North: $2.4M (up 15% YoY)
- South: $1.8M (down 5% YoY)
- East: $3.1M (up 22% YoY)
- West: $2.9M (up 8% YoY)

Product Categories:
- Electronics: 45% of sales (up from 38%)
- Home Goods: 30% of sales (down from 35%)
- Apparel: 25% of sales (stable)

Provide:
1. Key observations
2. Potential concerns
3. Recommended actions"""

    console.print(Panel(prompt, title="Sales Data", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": "You are a business analyst. Provide clear, actionable insights.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=1000,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-3 Analysis:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_creative_writing(client: OpenAI) -> None:
    """Demonstrate creative writing abilities."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Creative Writing\n")

    prompt = """Write a short story opening (150-200 words) with the following elements:
- Genre: Mystery/Thriller
- Setting: A lighthouse on a remote island
- Hook: Something unexpected has washed ashore
- Tone: Atmospheric and suspenseful"""

    console.print(Panel(prompt, title="Writing Prompt", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": "You are a talented fiction writer. Create engaging, "
                "atmospheric prose that draws readers in.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,  # Higher temperature for creativity
        max_tokens=400,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-3 Story Opening:[/bold green]\n")
    console.print(Panel(content, border_style="green"))


def demonstrate_explanation(client: OpenAI) -> None:
    """Demonstrate explanation capabilities."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Technical Explanation\n")

    prompt = """Explain how HTTPS works to someone with basic computer knowledge.
Cover:
1. What problem it solves
2. How encryption works (high level)
3. What the 'certificate' is for
4. Why the padlock icon matters

Keep it accessible but accurate, around 200-250 words."""

    console.print(Panel(prompt, title="Explanation Request", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": "You are a patient technical educator who makes complex "
                "topics accessible without oversimplifying.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-3 Explanation:[/bold green]\n")
    console.print(Panel(content, border_style="green"))


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Grok-3 Model Demonstration[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    demonstrate_general_capability(client)
    demonstrate_analysis(client)
    demonstrate_creative_writing(client)
    demonstrate_explanation(client)

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print("[dim]Grok-3 is a strong general-purpose model for diverse tasks.[/dim]\n")
