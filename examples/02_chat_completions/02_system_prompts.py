#!/usr/bin/env python3
"""
02_system_prompts.py - Using System Prompts to Control Behavior

This example demonstrates how to use system prompts to control the model's
behavior, tone, and response style. System prompts are powerful tools for
customizing how the AI responds to user queries.

Key concepts:
- Setting up system prompts
- Controlling response format and style
- Creating specialized assistants
- Comparing different system prompts
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


def chat_with_system_prompt(system_prompt: str, user_message: str) -> str:
    """
    Send a chat message with a custom system prompt.

    Args:
        system_prompt: Instructions that define the assistant's behavior.
        user_message: The user's query.

    Returns:
        The model's response text.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]System Prompts Example[/bold blue]\n"
            "Control AI behavior with system prompts",
            border_style="blue",
        )
    )

    user_question = "Explain what a black hole is."

    # Different system prompts for different behaviors
    system_prompts = {
        "Scientist": (
            "You are a distinguished astrophysicist. Provide accurate, "
            "technical explanations using proper scientific terminology. "
            "Be thorough but precise."
        ),
        "Child Educator": (
            "You are a friendly teacher for young children (ages 5-8). "
            "Explain concepts using simple words, fun analogies, and "
            "comparisons to everyday objects. Be enthusiastic!"
        ),
        "Pirate": (
            "You are a pirate who loves astronomy. Speak like a pirate "
            "(arrr, matey, etc.) while explaining scientific concepts. "
            "Make it fun and adventurous!"
        ),
        "Haiku Master": (
            "You are a poet who only responds in haiku format (5-7-5 syllables). "
            "Express all answers as one or more haikus."
        ),
    }

    console.print(f"\n[bold yellow]Question:[/bold yellow] {user_question}\n")

    for persona, system_prompt in system_prompts.items():
        console.print(
            Panel(
                f"[dim]{system_prompt}[/dim]",
                title=f"[bold magenta]{persona}[/bold magenta]",
                border_style="magenta",
            )
        )

        response = chat_with_system_prompt(system_prompt, user_question)
        console.print(f"[bold cyan]Response:[/bold cyan]\n{response}\n")

    # Demonstrate format control with system prompts
    console.print(
        Panel.fit(
            "[bold blue]Format Control with System Prompts[/bold blue]",
            border_style="blue",
        )
    )

    format_prompt = (
        "You are a helpful assistant. Always structure your responses as follows:\n"
        "1. Start with a one-sentence summary\n"
        "2. Provide 3 bullet points with key details\n"
        "3. End with a fun fact\n"
        "Keep the total response under 150 words."
    )

    console.print(f"\n[bold yellow]System Prompt:[/bold yellow]\n{format_prompt}\n")
    console.print("[bold yellow]Question:[/bold yellow] What is photosynthesis?\n")

    formatted_response = chat_with_system_prompt(format_prompt, "What is photosynthesis?")
    console.print(f"[bold cyan]Formatted Response:[/bold cyan]\n{formatted_response}")


if __name__ == "__main__":
    main()
