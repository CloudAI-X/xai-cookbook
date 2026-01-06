#!/usr/bin/env python3
"""
01_json_mode.py - Basic JSON Mode with response_format

This example demonstrates how to use JSON mode with xAI's chat completions API.
JSON mode ensures the model outputs valid JSON, making it easy to parse responses
programmatically.

Key concepts:
- Enabling JSON mode with response_format
- System prompts for JSON output guidance
- Parsing JSON responses safely
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


def basic_json_mode(prompt: str) -> dict:
    """
    Get a JSON response from the model using JSON mode.

    Args:
        prompt: The user's message to send to the model.

    Returns:
        Parsed JSON response as a dictionary.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that always outputs valid JSON. "
                "Respond only with JSON, no additional text.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)


def main():
    console.print(
        Panel.fit(
            "[bold blue]JSON Mode Example[/bold blue]\n"
            "Using response_format for guaranteed JSON output",
            border_style="blue",
        )
    )

    # Example 1: List of programming languages
    console.print("\n[bold yellow]Example 1: List Programming Languages[/bold yellow]")
    console.print(
        "[bold green]User:[/bold green] List 3 programming languages with their primary use cases"
    )

    result1 = basic_json_mode(
        "List 3 programming languages with their primary use cases. "
        "Return as JSON with a 'languages' array containing objects with "
        "'name' and 'use_case' fields."
    )

    console.print("[bold cyan]Response (parsed JSON):[/bold cyan]")
    syntax1 = Syntax(json.dumps(result1, indent=2), "json", theme="monokai")
    console.print(syntax1)

    # Access the data programmatically
    console.print("\n[dim]Programmatic access:[/dim]")
    for lang in result1.get("languages", []):
        console.print(f"  - {lang.get('name')}: {lang.get('use_case')}")

    # Example 2: Product information
    console.print("\n[bold yellow]Example 2: Extract Product Info[/bold yellow]")
    console.print("[bold green]User:[/bold green] Create a product entry for a laptop")

    result2 = basic_json_mode(
        "Create a product entry for a high-end laptop. Include fields: "
        "name, price (number), category, specifications (object with cpu, ram, storage), "
        "and available (boolean)."
    )

    console.print("[bold cyan]Response (parsed JSON):[/bold cyan]")
    syntax2 = Syntax(json.dumps(result2, indent=2), "json", theme="monokai")
    console.print(syntax2)

    # Example 3: Sentiment analysis
    console.print("\n[bold yellow]Example 3: Sentiment Analysis[/bold yellow]")
    console.print("[bold green]User:[/bold green] Analyze sentiment of a review")

    review_text = "The product exceeded my expectations! Great quality and fast shipping."

    result3 = basic_json_mode(
        f"Analyze the sentiment of this review: '{review_text}'. "
        "Return JSON with fields: sentiment (positive/negative/neutral), "
        "confidence (0-1), and key_phrases (array of strings)."
    )

    console.print("[bold cyan]Response (parsed JSON):[/bold cyan]")
    syntax3 = Syntax(json.dumps(result3, indent=2), "json", theme="monokai")
    console.print(syntax3)

    # Show the finish reason
    console.print("\n[bold yellow]Response Metadata:[/bold yellow]")
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": "Return a simple greeting as JSON."},
        ],
        response_format={"type": "json_object"},
    )

    console.print(f"  Finish Reason: {response.choices[0].finish_reason}")
    console.print(f"  Model: {response.model}")

    if response.usage:
        console.print(f"  Total Tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
