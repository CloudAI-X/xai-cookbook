#!/usr/bin/env python3
"""
01_basic_functions.py - Defining Basic Functions for Tool Calling

This example demonstrates how to define simple functions (tools) that the
xAI API can call. Function calling allows Grok to interact with external
systems, APIs, and perform actions based on user requests.

Key concepts:
- Defining function schemas with JSON Schema
- Required vs optional parameters
- Parameter types and descriptions
- Using enums for constrained values
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

# Define simple functions (tools) for the model to use
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g., 'San Francisco, CA' or 'Tokyo, Japan'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a specified timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The timezone, e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'",
                    },
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g., '2 + 2', '10 * 5'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_contacts",
            "description": "Search for contacts by name or email",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (name or email)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def demonstrate_tool_schema():
    """Display the tool schemas in a readable format."""
    console.print("\n[bold cyan]Tool Schemas:[/bold cyan]")

    for tool in TOOLS:
        func = tool["function"]
        console.print(f"\n[bold green]{func['name']}[/bold green]")
        console.print(f"  Description: {func['description']}")

        params = func["parameters"]["properties"]
        required = func["parameters"].get("required", [])

        console.print("  Parameters:")
        for param_name, param_info in params.items():
            req_marker = "[red]*[/red]" if param_name in required else ""
            param_type = param_info["type"]
            param_desc = param_info.get("description", "No description")

            if "enum" in param_info:
                param_type += f" (enum: {param_info['enum']})"

            console.print(f"    - {param_name}{req_marker}: {param_type}")
            console.print(f"      {param_desc}")


def basic_function_call(user_message: str) -> dict:
    """
    Send a message to Grok with tools available and get the response.

    Args:
        user_message: The user's message that may trigger a function call.

    Returns:
        Dictionary containing the response and any function call info.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": user_message}],
        tools=TOOLS,
        tool_choice="auto",  # Let the model decide whether to call a function
    )

    message = response.choices[0].message

    result = {
        "content": message.content,
        "tool_calls": [],
        "finish_reason": response.choices[0].finish_reason,
    }

    # Check if the model wants to call any functions
    if message.tool_calls:
        for tool_call in message.tool_calls:
            result["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "function_name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                }
            )

    return result


def main():
    console.print(
        Panel.fit(
            "[bold blue]Basic Function Definitions[/bold blue]\n"
            "Learn how to define tools for function calling",
            border_style="blue",
        )
    )

    # Show the tool schemas
    demonstrate_tool_schema()

    # Show raw JSON schema for one tool
    console.print("\n[bold yellow]Raw JSON Schema Example:[/bold yellow]")
    json_str = json.dumps(TOOLS[0], indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(syntax)

    # Test function calls with different prompts
    test_prompts = [
        "What's the weather like in Tokyo?",
        "What time is it in New York?",
        "Calculate 15 * 7 + 23",
        "Find contacts named John",
        "Tell me a joke",  # Should NOT trigger a function call
    ]

    console.print("\n[bold yellow]Testing Function Call Detection:[/bold yellow]")

    for prompt in test_prompts:
        console.print(f"\n[bold green]User:[/bold green] {prompt}")

        result = basic_function_call(prompt)

        if result["tool_calls"]:
            for call in result["tool_calls"]:
                console.print(f"  [cyan]Function called:[/cyan] {call['function_name']}")
                console.print(f"  [cyan]Arguments:[/cyan] {call['arguments']}")
        else:
            console.print("  [dim]No function called. Response:[/dim]")
            if result["content"]:
                console.print(f"  {result['content'][:100]}...")

    # Demonstrate tool_choice options
    console.print("\n[bold yellow]Tool Choice Options:[/bold yellow]")

    console.print('\n[cyan]tool_choice="auto"[/cyan] - Model decides (default)')
    console.print('[cyan]tool_choice="none"[/cyan] - Never call functions')
    console.print('[cyan]tool_choice="required"[/cyan] - Must call a function')
    console.print(
        '[cyan]tool_choice={"type": "function", "function": {"name": "get_weather"}}[/cyan]'
        " - Force specific function"
    )

    # Example: Force a specific function
    console.print("\n[bold green]Forcing get_weather function:[/bold green]")

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    if response.choices[0].message.tool_calls:
        call = response.choices[0].message.tool_calls[0]
        args = json.loads(call.function.arguments)
        console.print(f"  Function: {call.function.name}")
        console.print(f"  Arguments: {args}")


if __name__ == "__main__":
    main()
