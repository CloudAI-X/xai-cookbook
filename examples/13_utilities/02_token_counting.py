#!/usr/bin/env python3
"""
02_token_counting.py - Count Tokens Before Sending Requests

This example demonstrates practical techniques for counting tokens
before sending API requests to estimate costs and ensure requests
fit within context windows.

Key concepts:
- Pre-request token counting
- Cost estimation
- Context window management
- Batch processing optimization
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

# Model context windows and pricing
MODEL_SPECS = {
    "grok-4-1-fast-reasoning": {
        "context_window": 131072,
        "input_price_per_m": 0.30,
        "output_price_per_m": 0.50,
    },
    "grok-3": {
        "context_window": 131072,
        "input_price_per_m": 3.00,
        "output_price_per_m": 15.00,
    },
    "grok-4": {
        "context_window": 131072,
        "input_price_per_m": 2.00,
        "output_price_per_m": 10.00,
    },
    "grok-4-fast": {
        "context_window": 131072,
        "input_price_per_m": 5.00,
        "output_price_per_m": 25.00,
    },
}


def count_tokens(text: str, model: str = "grok-4-1-fast-reasoning") -> int:
    """
    Count tokens in text using the tokenize-text endpoint.

    Args:
        text: The text to tokenize.
        model: The model to use for tokenization.

    Returns:
        Number of tokens, or -1 on error.
    """
    url = f"{BASE_URL}/tokenize-text"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"text": text, "model": model}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return len(result.get("tokens", []))
    return -1


def estimate_message_tokens(messages: list[dict], model: str = "grok-4-1-fast-reasoning") -> dict:
    """
    Estimate tokens for a list of chat messages.

    Args:
        messages: List of message dictionaries with role and content.
        model: The model to use.

    Returns:
        Dictionary with token counts for each message and total.
    """
    results = {"messages": [], "total": 0}

    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")

        # Count content tokens
        token_count = count_tokens(content, model)

        # Add overhead for role formatting (approximate)
        overhead = 4  # Approximate tokens for role markers

        results["messages"].append(
            {
                "role": role,
                "content_tokens": token_count,
                "overhead": overhead,
                "total": token_count + overhead if token_count >= 0 else -1,
            }
        )

        if token_count >= 0:
            results["total"] += token_count + overhead

    return results


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "grok-4-1-fast-reasoning",
) -> dict:
    """
    Estimate the cost of a request.

    Args:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Estimated output tokens.
        model: The model to use.

    Returns:
        Dictionary with cost breakdown.
    """
    specs = MODEL_SPECS.get(model, MODEL_SPECS["grok-4-1-fast-reasoning"])

    input_cost = (input_tokens / 1_000_000) * specs["input_price_per_m"]
    output_cost = (output_tokens / 1_000_000) * specs["output_price_per_m"]

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def check_context_window(
    input_tokens: int,
    expected_output: int,
    model: str = "grok-4-1-fast-reasoning",
) -> dict:
    """
    Check if request fits within model's context window.

    Args:
        input_tokens: Number of input tokens.
        expected_output: Expected output tokens.
        model: The model to use.

    Returns:
        Dictionary with fit analysis.
    """
    specs = MODEL_SPECS.get(model, MODEL_SPECS["grok-4-1-fast-reasoning"])
    context_window = specs["context_window"]
    total_needed = input_tokens + expected_output

    return {
        "context_window": context_window,
        "input_tokens": input_tokens,
        "expected_output": expected_output,
        "total_needed": total_needed,
        "fits": total_needed <= context_window,
        "remaining": context_window - total_needed,
        "utilization": (total_needed / context_window) * 100,
    }


def batch_token_analysis(texts: list[str], model: str = "grok-4-1-fast-reasoning") -> list[dict]:
    """
    Analyze token counts for multiple texts.

    Args:
        texts: List of texts to analyze.
        model: The model to use.

    Returns:
        List of analysis results.
    """
    results = []

    for i, text in enumerate(texts):
        token_count = count_tokens(text, model)
        results.append(
            {
                "index": i,
                "text_length": len(text),
                "token_count": token_count,
                "chars_per_token": (len(text) / token_count if token_count > 0 else 0),
            }
        )

    return results


def main():
    console.print(
        Panel.fit(
            "[bold blue]Token Counting Example[/bold blue]\n"
            "Estimate tokens and costs before API calls",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Example 1: Count tokens in a prompt
    console.print("\n[bold yellow]Example 1: Count Prompt Tokens[/bold yellow]")

    prompt = """You are a helpful assistant. Please analyze the following code
and explain what it does:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

    token_count = count_tokens(prompt)
    console.print(f"[bold green]Prompt:[/bold green] [dim]{prompt[:100]}...[/dim]")
    console.print(f"[bold cyan]Token count:[/bold cyan] {token_count}")

    # Example 2: Estimate message tokens
    console.print("\n[bold yellow]Example 2: Multi-Message Token Count[/bold yellow]")

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What is a binary search tree?"},
        {
            "role": "assistant",
            "content": "A binary search tree (BST) is a data structure...",
        },
        {"role": "user", "content": "How do I implement one in Python?"},
    ]

    estimate = estimate_message_tokens(messages)

    msg_table = Table(show_header=True, header_style="bold cyan")
    msg_table.add_column("Role", style="green")
    msg_table.add_column("Content Preview")
    msg_table.add_column("Tokens", justify="right")

    for i, msg_info in enumerate(estimate["messages"]):
        content = messages[i]["content"][:30] + "..."
        msg_table.add_row(
            msg_info["role"],
            content,
            str(msg_info["total"]),
        )

    console.print(msg_table)
    console.print(f"[bold cyan]Total estimated tokens:[/bold cyan] {estimate['total']}")

    # Example 3: Cost estimation
    console.print("\n[bold yellow]Example 3: Cost Estimation[/bold yellow]")

    input_tokens = estimate["total"]
    expected_output = 500  # Estimate for response

    cost_table = Table(show_header=True, header_style="bold cyan")
    cost_table.add_column("Model", style="green")
    cost_table.add_column("Input Cost", justify="right")
    cost_table.add_column("Output Cost", justify="right")
    cost_table.add_column("Total Cost", justify="right")

    for model in ["grok-4-1-fast-reasoning", "grok-3", "grok-4"]:
        cost = estimate_cost(input_tokens, expected_output, model)
        cost_table.add_row(
            model,
            f"${cost['input_cost']:.6f}",
            f"${cost['output_cost']:.6f}",
            f"${cost['total_cost']:.6f}",
        )

    console.print(cost_table)
    console.print(f"[dim]Based on {input_tokens} input + {expected_output} output tokens[/dim]")

    # Example 4: Context window check
    console.print("\n[bold yellow]Example 4: Context Window Check[/bold yellow]")

    # Simulate a large prompt
    large_prompt = "Hello world. " * 1000
    large_token_count = count_tokens(large_prompt)

    for model in ["grok-4-1-fast-reasoning", "grok-4"]:
        check = check_context_window(large_token_count, 2000, model)

        status = "[green]FITS[/green]" if check["fits"] else "[red]TOO LARGE[/red]"
        console.print(
            f"\n[bold]{model}:[/bold] {status}\n"
            f"  Input: {check['input_tokens']:,} tokens\n"
            f"  Expected output: {check['expected_output']:,} tokens\n"
            f"  Total needed: {check['total_needed']:,} tokens\n"
            f"  Context window: {check['context_window']:,} tokens\n"
            f"  Utilization: {check['utilization']:.1f}%"
        )

    # Example 5: Batch analysis
    console.print("\n[bold yellow]Example 5: Batch Token Analysis[/bold yellow]")

    texts = [
        "Short message.",
        "A slightly longer message with more words.",
        "This is a medium-length paragraph that contains several sentences. "
        "It demonstrates how token counts scale with text length.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5,
    ]

    results = batch_token_analysis(texts)

    batch_table = Table(show_header=True, header_style="bold cyan")
    batch_table.add_column("#", style="green")
    batch_table.add_column("Text Length", justify="right")
    batch_table.add_column("Tokens", justify="right")
    batch_table.add_column("Chars/Token", justify="right")

    for r in results:
        batch_table.add_row(
            str(r["index"] + 1),
            str(r["text_length"]),
            str(r["token_count"]),
            f"{r['chars_per_token']:.1f}",
        )

    console.print(batch_table)

    # Best practices
    console.print("\n[bold yellow]Best Practices:[/bold yellow]")
    console.print(
        """
  [cyan]1. Pre-count for large requests:[/cyan]
     Always count tokens before sending large prompts to avoid
     unexpected costs or context window overflow.

  [cyan]2. Budget for output:[/cyan]
     Remember to reserve tokens for the model's response.
     A good rule: leave 20-30% of context for output.

  [cyan]3. Use max_tokens:[/cyan]
     Set max_tokens parameter to control output length and costs.

  [cyan]4. Batch wisely:[/cyan]
     For multiple items, calculate total tokens to optimize batching.

  [cyan]5. Cache token counts:[/cyan]
     If reusing prompts, cache token counts to reduce API calls.
"""
    )


if __name__ == "__main__":
    main()
