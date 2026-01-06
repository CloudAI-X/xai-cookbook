#!/usr/bin/env python3
"""
Grok-4-1 Fast Reasoning - Latest Generation Model

Model: grok-4-1-fast-reasoning
The newest iteration of xAI's fast reasoning model

Grok-4-1 Fast Reasoning features:
- Latest model improvements and optimizations
- Enhanced reasoning capabilities
- Improved instruction following
- Better handling of complex queries
- Updated knowledge cutoff

Best for: Applications requiring the latest model capabilities with reasoning
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


def demonstrate_advanced_reasoning(client: OpenAI) -> None:
    """Demonstrate the latest model's advanced reasoning."""
    console.print(
        Panel.fit(
            "[bold cyan]Grok-4-1 Fast Reasoning Demo[/bold cyan]\n"
            "Model: grok-4-1-fast-reasoning | Latest Generation",
            border_style="cyan",
        )
    )

    # Complex analytical task
    prompt = """Analyze this scenario using game theory:

Two competing ride-sharing companies are deciding whether to lower prices in a city.
- If both keep prices high: Each makes $10M profit
- If both lower prices: Each makes $6M profit
- If one lowers and one keeps high: The one lowering makes $12M, the other makes $4M

1. Identify this as a game theory problem type
2. Find the Nash equilibrium
3. Explain why the outcome might be suboptimal
4. Suggest mechanisms that could improve the outcome for both parties"""

    console.print("\n[bold]Task:[/bold] Game Theory Analysis\n")
    console.print(Panel(prompt, title="Scenario", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are an economics professor specializing in game theory. "
                "Provide rigorous analysis with clear explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4-1 Fast Reasoning Analysis:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))

    if response.usage:
        console.print(
            f"\n[dim]Tokens used - Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}[/dim]"
        )


def demonstrate_multi_step_problem(client: OpenAI) -> None:
    """Demonstrate multi-step problem solving."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Multi-Step Problem Solving\n")

    prompt = """Solve this optimization problem step by step:

A delivery company needs to design routes for 3 trucks serving 9 locations.
- Each truck can carry 100 units
- Demands at locations: A(30), B(25), C(40), D(35), E(20), F(45), G(15), H(30), I(25)
- Distances from depot (in km):
  A: 5, B: 8, C: 12, D: 6, E: 15, F: 10, G: 4, H: 9, I: 11

Requirements:
1. Each location must be served by exactly one truck
2. No truck can exceed capacity
3. Minimize total distance traveled

Find an efficient assignment of locations to trucks and calculate total distance."""

    console.print(Panel(prompt, title="Optimization Problem", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are an operations research expert. Solve optimization "
                "problems systematically, showing your methodology.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4-1 Fast Reasoning Solution:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_creative_reasoning(client: OpenAI) -> None:
    """Demonstrate creative problem-solving with reasoning."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Creative Problem Solving\n")

    prompt = """Design an innovative solution for this urban challenge:

A city of 500,000 people faces these interconnected problems:
1. Traffic congestion costing $200M annually in lost productivity
2. Air quality issues causing 1,000+ hospital visits per year
3. Limited public transit covering only 40% of the city
4. Aging infrastructure needing $500M in repairs

Budget: $300M over 5 years

Propose a creative, integrated solution that addresses multiple problems
simultaneously. Consider emerging technologies, behavioral economics,
and successful examples from other cities."""

    console.print(Panel(prompt, title="Urban Challenge", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are an urban planning innovation consultant. "
                "Combine analytical thinking with creative solutions.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Grok-4-1 Fast Reasoning Solution:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Grok-4-1 Fast Reasoning Demonstration[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    demonstrate_advanced_reasoning(client)
    demonstrate_multi_step_problem(client)
    demonstrate_creative_reasoning(client)

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print("[dim]Grok-4-1 Fast Reasoning is the latest generation model.[/dim]\n")
