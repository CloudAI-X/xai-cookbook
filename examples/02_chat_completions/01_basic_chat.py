#!/usr/bin/env python3
"""
01_basic_chat.py - Simple Chat Completion with xAI

This example demonstrates the most basic use of the xAI chat completions API.
It sends a single user message and displays the response.

Key concepts:
- Creating an OpenAI client configured for xAI
- Making a basic chat completion request
- Accessing response content and metadata
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


def basic_chat(prompt: str) -> str:
    """
    Send a simple chat message and get a response.

    Args:
        prompt: The user's message to send to the model.

    Returns:
        The model's response text.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Basic Chat Completion Example[/bold blue]\n"
            "Demonstrates the simplest xAI API call",
            border_style="blue",
        )
    )

    # Example 1: Simple question
    prompt1 = "What is the capital of France?"
    console.print(f"\n[bold green]User:[/bold green] {prompt1}")

    response1 = basic_chat(prompt1)
    console.print(f"[bold cyan]Grok:[/bold cyan] {response1}")

    # Example 2: Creative prompt
    prompt2 = "Write a haiku about programming."
    console.print(f"\n[bold green]User:[/bold green] {prompt2}")

    response2 = basic_chat(prompt2)
    console.print(f"[bold cyan]Grok:[/bold cyan] {response2}")

    # Example 3: Show full response object
    console.print("\n[bold yellow]Full Response Object Details:[/bold yellow]")

    full_response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": "Say hello!"}],
    )

    console.print(f"  Model: {full_response.model}")
    console.print(f"  ID: {full_response.id}")
    console.print(f"  Created: {full_response.created}")
    console.print(f"  Finish Reason: {full_response.choices[0].finish_reason}")

    if full_response.usage:
        console.print(f"  Prompt Tokens: {full_response.usage.prompt_tokens}")
        console.print(f"  Completion Tokens: {full_response.usage.completion_tokens}")
        console.print(f"  Total Tokens: {full_response.usage.total_tokens}")


if __name__ == "__main__":
    main()
