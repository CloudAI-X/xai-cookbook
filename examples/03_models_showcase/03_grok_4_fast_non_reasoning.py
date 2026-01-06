#!/usr/bin/env python3
"""
Grok-4 Fast Non-Reasoning - Speed-Optimized Extended Context

Model: grok-4-fast-non-reasoning
Context Window: 2M tokens (massive!)
Mode: Fast, direct responses without explicit reasoning chains

Grok-4 Fast Non-Reasoning excels at:
- Quick responses to straightforward queries
- Processing very long documents efficiently
- Summarization tasks
- Information extraction
- High-throughput applications

Best for: Long-context tasks prioritizing speed over detailed reasoning
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


def demonstrate_fast_summarization(client: OpenAI) -> None:
    """Demonstrate fast summarization without reasoning overhead."""
    console.print(
        Panel.fit(
            "[bold cyan]Grok-4 Fast Non-Reasoning Demo[/bold cyan]\n"
            "Model: grok-4-fast-non-reasoning | Context: 2M tokens | Speed-Optimized",
            border_style="cyan",
        )
    )

    # Long text summarization task
    article = """
The evolution of artificial intelligence has been marked by several paradigm shifts
over the past seven decades. Beginning with the symbolic AI approaches of the 1950s
and 1960s, researchers attempted to encode human knowledge explicitly into computer
systems through rules and logic. This approach, while elegant in theory, struggled
with the brittleness of hand-coded rules and the difficulty of capturing common-sense
knowledge.

The 1980s saw the rise of expert systems, which attempted to capture domain-specific
knowledge from human experts. These systems found commercial success in narrow
applications but again faced scalability challenges. The inability to learn from
data and adapt to new situations limited their broader applicability.

The statistical revolution of the 1990s and 2000s brought machine learning to the
forefront. Algorithms that could learn patterns from data began to outperform
hand-crafted rules in many domains. Support vector machines, random forests, and
other classical ML techniques enabled practical applications in spam detection,
recommendation systems, and basic image recognition.

The deep learning revolution, beginning around 2012 with AlexNet's breakthrough in
image classification, fundamentally changed the landscape. Neural networks with many
layers could learn hierarchical representations directly from raw data. This approach
scaled remarkably well with data and compute, leading to superhuman performance in
many perceptual tasks.

The emergence of large language models (LLMs) starting with GPT and BERT represented
another paradigm shift. Pre-training on vast text corpora followed by fine-tuning
enabled models to demonstrate broad capabilities across language tasks. The scaling
laws discovered by researchers showed that simply increasing model size, data, and
compute led to predictable improvements in capability.

Today, we stand at the frontier of artificial general intelligence research. Models
like GPT-4, Claude, and Grok demonstrate remarkable capabilities across reasoning,
coding, analysis, and creative tasks. The integration of multimodal understanding,
tool use, and long-context processing suggests we may be approaching systems with
increasingly general capabilities.
"""

    console.print("\n[bold]Task:[/bold] Fast Document Summarization\n")
    console.print(Panel(article[:500] + "...", title="Article (truncated)", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-fast-non-reasoning",
        messages=[
            {
                "role": "system",
                "content": "Provide concise, direct summaries without extensive preamble.",
            },
            {
                "role": "user",
                "content": f"Summarize this article in 3-4 bullet points:\n\n{article}",
            },
        ],
        temperature=0.3,
        max_tokens=500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Fast Summary:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))

    if response.usage:
        console.print(
            f"\n[dim]Tokens used - Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}[/dim]"
        )


def demonstrate_information_extraction(client: OpenAI) -> None:
    """Demonstrate fast information extraction."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Fast Information Extraction\n")

    document = """
MEETING NOTES - Q4 Planning Session
Date: December 15, 2024
Attendees: Sarah Chen (CEO), Mike Rodriguez (CTO), Lisa Park (CFO),
           James Wilson (VP Sales), Emily Brown (VP Marketing)

Key Decisions:
1. Approved $2.5M budget for AI infrastructure upgrade
2. Set Q1 2025 launch target for mobile app v2.0
3. Agreed to expand engineering team by 15 positions
4. Postponed European expansion to Q3 2025

Action Items:
- Mike: Finalize vendor selection for cloud migration by Dec 22
- James: Present updated sales projections by Dec 18
- Emily: Complete market research for new product line by Jan 5
- Lisa: Prepare board presentation for January meeting

Risk Discussion:
- Supply chain concerns may affect hardware delivery timelines
- Competitor XYZ announced similar product, need differentiation strategy
- Key engineer departure risk - retention bonuses approved

Next Meeting: January 8, 2025 at 2:00 PM
"""

    console.print(Panel(document, title="Meeting Notes", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-4-fast-non-reasoning",
        messages=[
            {
                "role": "system",
                "content": "Extract information precisely and format clearly.",
            },
            {
                "role": "user",
                "content": f"""Extract the following from these meeting notes in a structured format:
1. All action items with owners and deadlines
2. Budget decisions (amounts and purposes)
3. Key dates mentioned

Meeting Notes:
{document}""",
            },
        ],
        temperature=0.1,  # Very low for precise extraction
        max_tokens=800,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Extracted Information:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_quick_qa(client: OpenAI) -> None:
    """Demonstrate quick question-answering."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Rapid Q&A\n")

    questions = [
        "What is the capital of Japan?",
        "Convert 100 kilometers to miles.",
        "What year did World War II end?",
        "What is the chemical formula for water?",
        "Who wrote 'Pride and Prejudice'?",
    ]

    console.print("[bold]Answering multiple quick questions:[/bold]\n")

    for question in questions:
        response = client.chat.completions.create(
            model="grok-4-fast-non-reasoning",
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions directly and concisely in one sentence.",
                },
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=100,
        )

        answer = response.choices[0].message.content
        console.print(f"[yellow]Q:[/yellow] {question}")
        console.print(f"[green]A:[/green] {answer}\n")


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Grok-4 Fast Non-Reasoning Demonstration[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    demonstrate_fast_summarization(client)
    demonstrate_information_extraction(client)
    demonstrate_quick_qa(client)

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print("[dim]Grok-4 Fast Non-Reasoning is optimized for speed with 2M context.[/dim]\n")
