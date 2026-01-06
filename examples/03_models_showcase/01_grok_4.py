#!/usr/bin/env python3
"""
Grok-4 (grok-4-0709) - xAI's Flagship Model

Model: grok-4-0709
Context Window: 256K tokens
Pricing: $3/1M input, $15/1M output

Grok-4 is xAI's most capable model, excelling at:
- Complex reasoning and analysis
- Multi-step problem solving
- Nuanced understanding of context
- High-quality text generation
- Advanced coding tasks

Best for: Tasks requiring maximum capability and quality
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


def demonstrate_complex_reasoning(client: OpenAI) -> None:
    """Demonstrate Grok-4's complex reasoning capabilities."""
    console.print(
        Panel.fit(
            "[bold cyan]Grok-4 Flagship Model Demo[/bold cyan]\n"
            "Model: grok-4-0709 | Context: 256K | $3/$15 per 1M tokens",
            border_style="cyan",
        )
    )

    # Complex multi-step reasoning task
    prompt = """Analyze this business scenario and provide a strategic recommendation:

A mid-sized tech company (500 employees, $50M revenue) is considering two paths:
1. Acquire a smaller AI startup ($10M) with promising technology but no revenue
2. Invest the same amount in building an internal AI team over 2 years

Consider: market timing, talent acquisition challenges, integration risks,
competitive landscape, and long-term strategic value.

Provide a structured analysis with a clear recommendation."""

    console.print("\n[bold]Task:[/bold] Complex Business Strategy Analysis\n")
    console.print(Panel(prompt, title="Prompt", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-0709",
        messages=[
            {
                "role": "system",
                "content": "You are a senior strategic advisor with expertise in "
                "technology M&A and corporate strategy. Provide thorough, "
                "actionable analysis.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4 Analysis:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))

    # Display usage stats
    if response.usage:
        console.print(
            f"\n[dim]Tokens used - Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}[/dim]"
        )


def demonstrate_nuanced_understanding(client: OpenAI) -> None:
    """Demonstrate Grok-4's nuanced understanding."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Nuanced Ethical Analysis\n")

    prompt = """Consider this ethical dilemma in AI development:

A company has developed an AI system that can predict employee burnout with
85% accuracy by analyzing communication patterns. This could help prevent
mental health crises, but raises privacy concerns.

Analyze the ethical considerations from multiple stakeholder perspectives
(employees, employers, society) and suggest a balanced approach."""

    console.print(Panel(prompt, title="Ethical Dilemma", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-0709",
        messages=[
            {
                "role": "system",
                "content": "You are an AI ethics expert. Provide balanced, "
                "thoughtful analysis considering multiple perspectives.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4 Ethical Analysis:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Grok-4 Flagship Model Demonstration[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    demonstrate_complex_reasoning(client)
    demonstrate_nuanced_understanding(client)

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print("[dim]Grok-4 is ideal for tasks requiring maximum capability.[/dim]\n")
