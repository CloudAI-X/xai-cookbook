#!/usr/bin/env python3
"""
01_tokenization.py - Tokenize Text with xAI API

This example demonstrates how to use the /v1/tokenize-text endpoint
to tokenize text and understand token usage before sending requests.

Key concepts:
- What are tokens
- Using the tokenize-text endpoint
- Token visualization
- Model-specific tokenization
"""

import os

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

# xAI API configuration
API_KEY = os.environ.get("X_AI_API_KEY")
BASE_URL = "https://api.x.ai/v1"


def tokenize_text(text: str, model: str = "grok-4-1-fast-reasoning") -> dict:
    """
    Tokenize text using the xAI API.

    Args:
        text: The text to tokenize.
        model: The model to use for tokenization.

    Returns:
        Dictionary containing tokens and metadata.
    """
    url = f"{BASE_URL}/tokenize-text"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "text": text,
        "model": model,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def visualize_tokens(tokens: list[int], text: str) -> None:
    """
    Visualize how text is split into tokens.

    Note: This is an approximation since we don't have the actual
    token-to-text mapping from the API.
    """
    console.print(f"\n[bold cyan]Token IDs ({len(tokens)} tokens):[/bold cyan]")

    # Display token IDs in rows
    row_size = 10
    for i in range(0, len(tokens), row_size):
        row = tokens[i : i + row_size]
        console.print(f"  {row}")


def estimate_cost(token_count: int, rate_per_million: float) -> float:
    """Calculate estimated cost based on token count."""
    return (token_count / 1_000_000) * rate_per_million


def main():
    console.print(
        Panel.fit(
            "[bold blue]Tokenization Example[/bold blue]\n"
            "Understand token usage before sending requests",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # What are tokens
    console.print("\n[bold yellow]What are Tokens?[/bold yellow]")
    console.print(
        """
  Tokens are the [cyan]basic units[/cyan] that language models use to process text.
  They can be:

  - Whole words: "hello" -> 1 token
  - Parts of words: "understanding" -> 2-3 tokens
  - Punctuation: "!" -> 1 token
  - Spaces: Sometimes included in tokens

  [dim]Token count affects both pricing and context window usage.[/dim]
"""
    )

    # Example 1: Simple tokenization
    console.print("\n[bold yellow]Example 1: Simple Text[/bold yellow]")

    simple_text = "Hello, world! How are you today?"
    console.print(f'[bold green]Text:[/bold green] "{simple_text}"')

    result = tokenize_text(simple_text)

    if "error" not in result:
        tokens = result.get("tokens", [])
        console.print(f"[bold cyan]Token count:[/bold cyan] {len(tokens)}")
        visualize_tokens(tokens, simple_text)
    else:
        console.print(f"[red]Error:[/red] {result.get('message')}")

    # Example 2: Complex text
    console.print("\n[bold yellow]Example 2: Complex Text[/bold yellow]")

    complex_text = """
    Machine learning is a subset of artificial intelligence (AI) that enables
    systems to learn and improve from experience without being explicitly
    programmed. It focuses on developing algorithms that can access data
    and use it to learn for themselves.
    """
    console.print(f"[bold green]Text:[/bold green] [dim]{complex_text.strip()[:100]}...[/dim]")

    result = tokenize_text(complex_text)

    if "error" not in result:
        tokens = result.get("tokens", [])
        console.print(f"[bold cyan]Token count:[/bold cyan] {len(tokens)}")
        console.print(f"[dim]Characters: {len(complex_text)}, Tokens: {len(tokens)}")
        if len(tokens) > 0:
            console.print(f"Ratio: ~{len(complex_text) / len(tokens):.1f} chars/token[/dim]")
        else:
            console.print("[dim]Ratio: N/A (tokenization API may be unavailable)[/dim]")
    else:
        console.print(f"[red]Error:[/red] {result.get('message')}")

    # Example 3: Code tokenization
    console.print("\n[bold yellow]Example 3: Code Tokenization[/bold yellow]")

    code_text = '''def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    console.print("[bold green]Code:[/bold green]")
    console.print(f"[dim]{code_text}[/dim]")

    result = tokenize_text(code_text)

    if "error" not in result:
        tokens = result.get("tokens", [])
        console.print(f"[bold cyan]Token count:[/bold cyan] {len(tokens)}")
    else:
        console.print(f"[red]Error:[/red] {result.get('message')}")

    # Example 4: Different languages
    console.print("\n[bold yellow]Example 4: Multi-language Text[/bold yellow]")

    texts = {
        "English": "Hello, how are you?",
        "Spanish": "Hola, como estas?",
        "Japanese": "Hello, how are you?",  # Placeholder
        "Emoji": "I love coding! 100",
    }

    results_table = Table(show_header=True, header_style="bold cyan")
    results_table.add_column("Language", style="green")
    results_table.add_column("Text")
    results_table.add_column("Tokens", justify="right")
    results_table.add_column("Chars/Token", justify="right")

    for lang, text in texts.items():
        result = tokenize_text(text)
        if "error" not in result:
            tokens = result.get("tokens", [])
            token_count = len(tokens)
            ratio = len(text) / token_count if token_count > 0 else 0
            results_table.add_row(lang, text, str(token_count), f"{ratio:.1f}")
        else:
            results_table.add_row(lang, text, "Error", "-")

    console.print(results_table)

    # Cost estimation
    console.print("\n[bold yellow]Cost Estimation:[/bold yellow]")

    # Pricing table (approximate - check docs.x.ai for current pricing)
    pricing_table = Table(show_header=True, header_style="bold cyan")
    pricing_table.add_column("Model", style="green")
    pricing_table.add_column("Input ($/M tokens)")
    pricing_table.add_column("Output ($/M tokens)")

    pricing_table.add_row("grok-4-1-fast-reasoning", "$0.30", "$0.50")
    pricing_table.add_row("grok-3", "$3.00", "$15.00")
    pricing_table.add_row("grok-4", "$2.00", "$10.00")
    pricing_table.add_row("grok-4-fast", "$5.00", "$25.00")

    console.print(pricing_table)
    console.print("[dim]Note: Check docs.x.ai for current pricing[/dim]")

    # API reference
    console.print("\n[bold yellow]API Reference:[/bold yellow]")
    console.print(
        """
[dim]POST https://api.x.ai/v1/tokenize-text

Headers:
  Authorization: Bearer {api_key}
  Content-Type: application/json

Body:
{
  "text": "Your text here",
  "model": "grok-4-1-fast-reasoning"  // Optional, defaults to model's tokenizer
}

Response:
{
  "tokens": [1234, 5678, ...],  // Array of token IDs
  "count": 10                   // Total token count
}[/dim]
"""
    )

    # Tips
    console.print("\n[bold yellow]Tokenization Tips:[/bold yellow]")
    console.print(
        """
  1. [cyan]Estimate before sending:[/cyan]
     Use tokenize-text to estimate costs before large requests

  2. [cyan]Context window:[/cyan]
     Total tokens (prompt + response) must fit in context window
     - grok-4-1-fast-reasoning: 131,072 tokens
     - grok-4: 131,072 tokens

  3. [cyan]Efficient prompts:[/cyan]
     - Be concise but clear
     - Avoid unnecessary repetition
     - Use structured formats when appropriate

  4. [cyan]Token variance:[/cyan]
     - Different models may tokenize differently
     - Special characters and code often use more tokens
     - Non-English text may use more tokens per word
"""
    )


if __name__ == "__main__":
    main()
