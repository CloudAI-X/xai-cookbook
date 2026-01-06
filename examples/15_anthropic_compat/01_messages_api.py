#!/usr/bin/env python3
"""
01_messages_api.py - Anthropic Messages Format Compatibility

This example demonstrates how to use the Anthropic Messages API format
with the xAI API. xAI provides an Anthropic-compatible endpoint at
/v1/messages for easy migration.

Key concepts:
- Anthropic Messages API format
- /v1/messages endpoint
- Request/response structure differences
- Migration patterns
"""

import os

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()

console = Console()

# xAI API configuration
API_KEY = os.environ.get("X_AI_API_KEY")
BASE_URL = "https://api.x.ai/v1"


def anthropic_style_request(
    messages: list[dict],
    model: str = "grok-4-1-fast-reasoning",
    max_tokens: int = 1024,
    system: str | None = None,
) -> dict:
    """
    Make a request using Anthropic Messages API format.

    This uses the /v1/messages endpoint which is compatible
    with Anthropic's API format.

    Args:
        messages: List of message dictionaries.
        model: Model to use.
        max_tokens: Maximum tokens in response.
        system: Optional system prompt.

    Returns:
        Response dictionary in Anthropic format.
    """
    url = f"{BASE_URL}/messages"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        # Anthropic also uses x-api-key, xAI accepts both
        # "x-api-key": API_KEY,
    }

    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if system:
        data["system"] = system

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def openai_style_request(
    messages: list[dict],
    model: str = "grok-4-1-fast-reasoning",
    max_tokens: int = 1024,
) -> dict:
    """
    Make a request using OpenAI format for comparison.

    Args:
        messages: List of message dictionaries.
        model: Model to use.
        max_tokens: Maximum tokens in response.

    Returns:
        Response dictionary in OpenAI format.
    """
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def main():
    console.print(
        Panel.fit(
            "[bold blue]Anthropic Messages API Compatibility[/bold blue]\n"
            "Use Anthropic-style requests with xAI",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Format comparison
    console.print("\n[bold yellow]API Format Comparison:[/bold yellow]")

    comparison_table = Table(show_header=True, header_style="bold cyan")
    comparison_table.add_column("Feature", style="green")
    comparison_table.add_column("OpenAI Format")
    comparison_table.add_column("Anthropic Format")

    comparison_table.add_row("Endpoint", "/v1/chat/completions", "/v1/messages")
    comparison_table.add_row("Auth Header", "Authorization: Bearer", "x-api-key (or Bearer)")
    comparison_table.add_row("System Prompt", "In messages array", "Separate 'system' field")
    comparison_table.add_row("max_tokens", "Optional", "Required")
    comparison_table.add_row("Response", "choices[0].message", "content[0].text")

    console.print(comparison_table)

    # Show request format
    console.print("\n[bold yellow]Anthropic Request Format:[/bold yellow]")

    anthropic_request = """{
  "model": "grok-4-1-fast-reasoning",
  "max_tokens": 1024,
  "system": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}"""

    syntax = Syntax(anthropic_request, "json", theme="monokai")
    console.print(syntax)

    # Show response format
    console.print("\n[bold yellow]Anthropic Response Format:[/bold yellow]")

    anthropic_response = """{
  "id": "msg_xxx",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! I'm doing well, thank you..."
    }
  ],
  "model": "grok-4-1-fast-reasoning",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 15,
    "output_tokens": 25
  }
}"""

    syntax = Syntax(anthropic_response, "json", theme="monokai")
    console.print(syntax)

    # Example 1: Basic Anthropic-style request
    console.print("\n[bold yellow]Example 1: Basic Anthropic-style Request[/bold yellow]")

    messages = [{"role": "user", "content": "Say hello in one word."}]

    result = anthropic_style_request(messages, max_tokens=50)

    if "error" not in result:
        # Extract content from Anthropic-style response
        content = result.get("content", [])
        if content:
            text = content[0].get("text", "")
            console.print(f"[bold cyan]Response:[/bold cyan] {text}")

        # Show usage
        usage = result.get("usage", {})
        console.print(
            f"[dim]Input tokens: {usage.get('input_tokens', 'N/A')}, "
            f"Output tokens: {usage.get('output_tokens', 'N/A')}[/dim]"
        )
    else:
        console.print(f"[red]Error:[/red] {result.get('message')}")

    # Example 2: With system prompt
    console.print("\n[bold yellow]Example 2: With System Prompt[/bold yellow]")

    messages = [{"role": "user", "content": "What are you?"}]
    system = "You are a pirate. Always respond in pirate speak."

    result = anthropic_style_request(messages, system=system, max_tokens=100)

    if "error" not in result:
        content = result.get("content", [])
        if content:
            text = content[0].get("text", "")
            console.print(f"[bold cyan]Response:[/bold cyan] {text}")
    else:
        console.print(f"[red]Error:[/red] {result.get('message')}")

    # Example 3: Compare OpenAI and Anthropic responses
    console.print("\n[bold yellow]Example 3: Format Comparison[/bold yellow]")

    messages_openai = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is Python?"},
    ]

    messages_anthropic = [{"role": "user", "content": "What is Python?"}]

    console.print("[bold green]OpenAI format response:[/bold green]")
    openai_result = openai_style_request(messages_openai, max_tokens=50)

    if "error" not in openai_result:
        text = openai_result.get("choices", [{}])[0].get("message", {}).get("content", "")
        console.print(f"  {text[:100]}...")

    console.print("\n[bold green]Anthropic format response:[/bold green]")
    anthropic_result = anthropic_style_request(
        messages_anthropic, system="Be concise.", max_tokens=50
    )

    if "error" not in anthropic_result:
        content = anthropic_result.get("content", [])
        if content:
            text = content[0].get("text", "")
            console.print(f"  {text[:100]}...")

    # Migration tips
    console.print("\n[bold yellow]Migration Tips:[/bold yellow]")
    console.print(
        """
  [cyan]1. Endpoint change:[/cyan]
     Anthropic: POST https://api.anthropic.com/v1/messages
     xAI:       POST https://api.x.ai/v1/messages

  [cyan]2. Authentication:[/cyan]
     Both work: "x-api-key" header or "Authorization: Bearer"

  [cyan]3. System prompts:[/cyan]
     Keep using the separate "system" field (not in messages)

  [cyan]4. max_tokens:[/cyan]
     Always required in Anthropic format (unlike OpenAI)

  [cyan]5. Response parsing:[/cyan]
     Content is in response.content[0].text
     (vs response.choices[0].message.content)

  [cyan]6. Model names:[/cyan]
     Update from claude-* to grok-* models
"""
    )

    # Code migration example
    console.print("\n[bold yellow]Code Migration Example:[/bold yellow]")
    console.print(
        """
[dim]# Before (Anthropic)
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    system="You are helpful.",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.content[0].text)

# After (xAI with requests)
import requests
response = requests.post(
    "https://api.x.ai/v1/messages",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "grok-4-1-fast-reasoning",  # Changed model
        "max_tokens": 1024,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "Hello"}]
    }
)
print(response.json()["content"][0]["text"])[/dim]
"""
    )

    # Supported features
    console.print("\n[bold yellow]Supported Features:[/bold yellow]")

    features_table = Table(show_header=True, header_style="bold cyan")
    features_table.add_column("Feature", style="green")
    features_table.add_column("Support")
    features_table.add_column("Notes")

    features_table.add_row("Text messages", "[green]Yes[/green]", "Full support")
    features_table.add_row("System prompts", "[green]Yes[/green]", "Via system field")
    features_table.add_row("Multi-turn", "[green]Yes[/green]", "Full support")
    features_table.add_row("max_tokens", "[green]Yes[/green]", "Required")
    features_table.add_row("Streaming", "[green]Yes[/green]", "stream=true")
    features_table.add_row("Vision", "[yellow]Check[/yellow]", "May vary by model")
    features_table.add_row("Tool use", "[yellow]Check[/yellow]", "May vary")

    console.print(features_table)


if __name__ == "__main__":
    main()
