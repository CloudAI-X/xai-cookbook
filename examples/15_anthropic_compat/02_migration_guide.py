#!/usr/bin/env python3
"""
02_migration_guide.py - Migrating from Anthropic SDK

This example provides a comprehensive guide for migrating applications
from the Anthropic SDK to the xAI API, covering common patterns and
code transformations.

Key concepts:
- SDK migration patterns
- Code transformation examples
- Handling differences
- Testing migration
"""

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

# xAI API configuration
API_KEY = os.environ.get("X_AI_API_KEY")


def main():
    console.print(
        Panel.fit(
            "[bold blue]Migration Guide: Anthropic to xAI[/bold blue]\n"
            "Step-by-step migration assistance",
            border_style="blue",
        )
    )

    # Migration overview
    console.print("\n[bold yellow]Migration Overview:[/bold yellow]")
    console.print(
        """
  Migrating from Anthropic to xAI involves:

  1. [cyan]API endpoint change[/cyan] - Update base URL
  2. [cyan]Model name updates[/cyan] - Use Grok models
  3. [cyan]Response parsing[/cyan] - Minor structure differences
  4. [cyan]Feature mapping[/cyan] - Ensure feature availability
"""
    )

    # Step 1: Dependencies
    console.print("\n[bold yellow]Step 1: Update Dependencies[/bold yellow]")
    console.print(
        """
  [cyan]Option A: Use OpenAI SDK (Recommended)[/cyan]
  [dim]# Remove anthropic, add openai
  pip uninstall anthropic
  pip install openai[/dim]

  [cyan]Option B: Use requests directly[/cyan]
  [dim]pip install requests[/dim]

  [cyan]Option C: Keep anthropic SDK (limited compatibility)[/cyan]
  [dim]# May work with custom base_url, but not recommended[/dim]
"""
    )

    # Step 2: Client initialization
    console.print("\n[bold yellow]Step 2: Update Client Initialization[/bold yellow]")
    console.print(
        """
[dim]# Before (Anthropic SDK)
import anthropic
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# After (OpenAI SDK with xAI)
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("X_AI_API_KEY"),
    base_url="https://api.x.ai/v1"
)[/dim]
"""
    )

    # Step 3: Model mapping
    console.print("\n[bold yellow]Step 3: Map Models[/bold yellow]")

    model_table = Table(show_header=True, header_style="bold cyan")
    model_table.add_column("Anthropic Model", style="green")
    model_table.add_column("xAI Equivalent")
    model_table.add_column("Notes")

    model_table.add_row("claude-3-5-sonnet", "grok-4-fast", "Fast, capable")
    model_table.add_row("claude-3-opus", "grok-4", "Most capable")
    model_table.add_row("claude-3-sonnet", "grok-3", "Balanced")
    model_table.add_row("claude-3-haiku", "grok-4-1-fast-reasoning", "Fast, efficient")

    console.print(model_table)

    # Step 4: API call conversion
    console.print("\n[bold yellow]Step 4: Convert API Calls[/bold yellow]")
    console.print(
        """
[dim]# Before (Anthropic SDK)
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
text = response.content[0].text

# After (OpenAI SDK with xAI)
response = client.chat.completions.create(
    model="grok-3",
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
text = response.choices[0].message.content[/dim]
"""
    )

    # Step 5: Response parsing
    console.print("\n[bold yellow]Step 5: Update Response Parsing[/bold yellow]")

    parsing_table = Table(show_header=True, header_style="bold cyan")
    parsing_table.add_column("Data", style="green")
    parsing_table.add_column("Anthropic")
    parsing_table.add_column("OpenAI/xAI")

    parsing_table.add_row(
        "Message text",
        "response.content[0].text",
        "response.choices[0].message.content",
    )
    parsing_table.add_row(
        "Model used",
        "response.model",
        "response.model",
    )
    parsing_table.add_row(
        "Input tokens",
        "response.usage.input_tokens",
        "response.usage.prompt_tokens",
    )
    parsing_table.add_row(
        "Output tokens",
        "response.usage.output_tokens",
        "response.usage.completion_tokens",
    )
    parsing_table.add_row(
        "Stop reason",
        "response.stop_reason",
        "response.choices[0].finish_reason",
    )

    console.print(parsing_table)

    # Step 6: Streaming
    console.print("\n[bold yellow]Step 6: Update Streaming Code[/bold yellow]")
    console.print(
        """
[dim]# Before (Anthropic SDK)
with client.messages.stream(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")

# After (OpenAI SDK with xAI)
stream = client.chat.completions.create(
    model="grok-3",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")[/dim]
"""
    )

    # Step 7: Tool/Function calling
    console.print("\n[bold yellow]Step 7: Update Tool/Function Calls[/bold yellow]")
    console.print(
        """
[dim]# Before (Anthropic tools)
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    tools=[{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }],
    messages=[{"role": "user", "content": "Weather in NYC?"}]
)

# After (OpenAI tools)
response = client.chat.completions.create(
    model="grok-3",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }],
    messages=[{"role": "user", "content": "Weather in NYC?"}]
)[/dim]
"""
    )

    # Step 8: Error handling
    console.print("\n[bold yellow]Step 8: Update Error Handling[/bold yellow]")
    console.print(
        """
[dim]# Before (Anthropic exceptions)
from anthropic import (
    APIError,
    RateLimitError,
    AuthenticationError
)

try:
    response = client.messages.create(...)
except RateLimitError:
    # Handle rate limit
except AuthenticationError:
    # Handle auth error
except APIError as e:
    # Handle other errors

# After (OpenAI exceptions)
from openai import (
    APIError,
    RateLimitError,
    AuthenticationError
)

try:
    response = client.chat.completions.create(...)
except RateLimitError:
    # Handle rate limit
except AuthenticationError:
    # Handle auth error
except APIError as e:
    # Handle other errors[/dim]
"""
    )

    # Migration checklist
    console.print("\n[bold yellow]Migration Checklist:[/bold yellow]")
    console.print(
        """
  [ ] Update dependencies (anthropic -> openai)
  [ ] Change API key environment variable
  [ ] Update client initialization with xAI base_url
  [ ] Replace claude-* model names with grok-*
  [ ] Convert messages.create() to chat.completions.create()
  [ ] Move system prompt into messages array
  [ ] Update response parsing (content[0].text -> choices[0].message.content)
  [ ] Update token usage field names
  [ ] Convert streaming code
  [ ] Update tool definitions (input_schema -> parameters)
  [ ] Update error handling imports
  [ ] Test all functionality
  [ ] Update documentation
"""
    )

    # Common issues
    console.print("\n[bold yellow]Common Migration Issues:[/bold yellow]")

    issues_table = Table(show_header=True, header_style="bold cyan")
    issues_table.add_column("Issue", style="green")
    issues_table.add_column("Cause")
    issues_table.add_column("Solution")

    issues_table.add_row(
        "401 Unauthorized",
        "Wrong API key",
        "Use X_AI_API_KEY, not ANTHROPIC_API_KEY",
    )
    issues_table.add_row(
        "Model not found",
        "Using claude-* models",
        "Change to grok-* models",
    )
    issues_table.add_row(
        "Response parsing error",
        "Different response structure",
        "Use choices[0].message.content",
    )
    issues_table.add_row(
        "Missing max_tokens",
        "Required in Anthropic, optional in OpenAI",
        "Add max_tokens parameter",
    )

    console.print(issues_table)

    # Testing strategy
    console.print("\n[bold yellow]Testing Strategy:[/bold yellow]")
    console.print(
        """
  [cyan]1. Unit Tests:[/cyan]
     - Mock API responses with expected format
     - Test response parsing
     - Verify error handling

  [cyan]2. Integration Tests:[/cyan]
     - Test against actual xAI API
     - Compare outputs with previous Anthropic outputs
     - Verify functionality parity

  [cyan]3. Gradual Rollout:[/cyan]
     - Start with non-critical features
     - A/B test responses
     - Monitor for regressions

  [cyan]4. Fallback Strategy:[/cyan]
     - Keep Anthropic code available
     - Use feature flags for switching
     - Have rollback plan ready
"""
    )

    # Summary
    console.print("\n[bold yellow]Summary:[/bold yellow]")
    console.print(
        """
  Migration from Anthropic to xAI is straightforward:

  1. The OpenAI SDK works directly with xAI
  2. Main changes are model names and response parsing
  3. Most features have direct equivalents
  4. Test thoroughly before full migration

  [cyan]Resources:[/cyan]
  - xAI Docs: https://docs.x.ai
  - OpenAI SDK: https://github.com/openai/openai-python
  - xAI Console: https://console.x.ai
"""
    )

    # Status
    console.print("\n[bold yellow]Your Setup:[/bold yellow]")
    if API_KEY:
        console.print("[green]X_AI_API_KEY is configured.[/green] Ready to migrate!")
    else:
        console.print("[yellow]X_AI_API_KEY not set.[/yellow] Set it before testing.")


if __name__ == "__main__":
    main()
