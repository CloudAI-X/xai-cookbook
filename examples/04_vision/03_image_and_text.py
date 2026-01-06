#!/usr/bin/env python3
"""
03_image_and_text.py - Mixed Content Analysis (Text + Images)

This example demonstrates how to combine text context with image analysis
for more sophisticated understanding. Grok can use provided text context
to better interpret images or analyze how images relate to text content.

Key concepts:
- Providing text context for image analysis
- Document analysis with images and text
- Multi-modal reasoning combining visual and textual information
"""

import os

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

# Sample image for mixed content analysis
SAMPLE_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/800px-Tour_Eiffel_Wikimedia_Commons.jpg"


def analyze_with_context(image_url: str, context: str, question: str) -> str:
    """
    Analyze an image with additional text context.

    Args:
        image_url: URL of the image to analyze.
        context: Background text information to consider.
        question: Specific question to answer about the image.

    Returns:
        The model's contextual analysis.
    """
    combined_prompt = f"""Context Information:
{context}

Question: {question}

Please analyze the image considering the context provided above."""

    response = client.chat.completions.create(
        model="grok-4-1-fast-non-reasoning",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def multi_turn_vision_chat(image_url: str, conversation: list[dict]) -> str:
    """
    Have a multi-turn conversation about an image.

    Args:
        image_url: URL of the image being discussed.
        conversation: List of conversation messages.

    Returns:
        The model's latest response.
    """
    # Add the image to the first user message
    messages = []
    for i, msg in enumerate(conversation):
        if i == 0 and msg["role"] == "user":
            # First user message includes the image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            )
        else:
            messages.append(msg)

    response = client.chat.completions.create(
        model="grok-4-1-fast-non-reasoning",
        messages=messages,
    )
    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Mixed Content Analysis (Text + Images)[/bold blue]\n"
            "Combine text context with image analysis for richer understanding",
            border_style="blue",
        )
    )

    console.print(f"\n[dim]Using image: {SAMPLE_IMAGE}[/dim]")

    # Example 1: Image analysis with historical context
    console.print("\n[bold yellow]Example 1: Analysis with Historical Context[/bold yellow]")

    historical_context = """
    The Eiffel Tower was built between 1887 and 1889 as the entrance arch
    for the 1889 World's Fair. It was initially criticized by some of France's
    leading artists and intellectuals for its design, but has become a global
    cultural icon of France and one of the most recognizable structures in the world.
    The tower is 330 meters (1,083 ft) tall and was the tallest man-made structure
    in the world for 41 years until the Chrysler Building was built in 1930.
    """

    analysis1 = analyze_with_context(
        SAMPLE_IMAGE,
        historical_context,
        "Based on the historical context provided, what elements of the image "
        "demonstrate the tower's significance and architectural innovation?",
    )
    console.print(Panel(analysis1, title="Historical Context Analysis", border_style="green"))

    # Example 2: Technical analysis with specifications
    console.print("\n[bold yellow]Example 2: Technical Analysis with Specs[/bold yellow]")

    technical_context = """
    The Eiffel Tower specifications:
    - Total height: 330 meters (with antennas)
    - Weight: 10,100 tons
    - Made of: Puddled iron (wrought iron)
    - Number of rivets: 2.5 million
    - Number of iron pieces: 18,038
    - Three observation levels at 57m, 115m, and 276m
    """

    analysis2 = analyze_with_context(
        SAMPLE_IMAGE,
        technical_context,
        "Looking at this image, can you identify any of the technical features "
        "mentioned in the specifications? What structural elements are visible?",
    )
    console.print(Panel(analysis2, title="Technical Analysis", border_style="cyan"))

    # Example 3: Multi-turn conversation about an image
    console.print("\n[bold yellow]Example 3: Multi-turn Vision Conversation[/bold yellow]")

    # Build conversation turn by turn
    conversation = [
        {"role": "user", "content": "What's the main subject of this image?"},
    ]

    response1 = multi_turn_vision_chat(SAMPLE_IMAGE, conversation)
    console.print(f"[bold green]User:[/bold green] {conversation[0]['content']}")
    console.print(f"[bold cyan]Grok:[/bold cyan] {response1}\n")

    # Continue the conversation
    conversation.append({"role": "assistant", "content": response1})
    conversation.append(
        {"role": "user", "content": "What time of day does this appear to be taken?"}
    )

    response2 = multi_turn_vision_chat(SAMPLE_IMAGE, conversation)
    console.print(f"[bold green]User:[/bold green] {conversation[-1]['content']}")
    console.print(f"[bold cyan]Grok:[/bold cyan] {response2}\n")

    # One more turn
    conversation.append({"role": "assistant", "content": response2})
    conversation.append(
        {
            "role": "user",
            "content": "Based on what you can see, what might be a good activity to do there?",
        }
    )

    response3 = multi_turn_vision_chat(SAMPLE_IMAGE, conversation)
    console.print(f"[bold green]User:[/bold green] {conversation[-1]['content']}")
    console.print(f"[bold cyan]Grok:[/bold cyan] {response3}")

    console.print(
        "\n[green]Mixed content analysis complete![/green] "
        "Combining text and images enables richer, more contextual understanding."
    )


if __name__ == "__main__":
    main()
