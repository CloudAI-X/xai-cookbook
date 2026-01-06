#!/usr/bin/env python3
"""
02_function_execution.py - Executing Function Calls and Responding

This example demonstrates the complete function calling flow:
1. Send a message with tools defined
2. Model requests a function call
3. Execute the function locally
4. Send the result back to the model
5. Get the final response

Key concepts:
- Implementing actual function handlers
- Processing tool calls from responses
- Sending function results back to the model
- Multi-turn conversation with function calls
"""

import json
import os
from datetime import datetime

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

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'Tokyo' or 'New York'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g., 'AAPL', 'GOOGL'",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]


# Simulated function implementations
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulate getting weather data."""
    # In a real app, this would call a weather API
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
        "New York": {"temp": 18, "condition": "Sunny", "humidity": 45},
        "London": {"temp": 15, "condition": "Rainy", "humidity": 80},
        "Sydney": {"temp": 25, "condition": "Clear", "humidity": 55},
    }

    # Default weather for unknown locations
    data = weather_data.get(location, {"temp": 20, "condition": "Unknown", "humidity": 50})

    temp = data["temp"]
    if unit == "fahrenheit":
        temp = (temp * 9 / 5) + 32

    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
        "humidity": data["humidity"],
    }


def get_stock_price(symbol: str) -> dict:
    """Simulate getting stock price data."""
    # In a real app, this would call a stock API
    stock_data = {
        "AAPL": {"price": 178.50, "change": 2.30, "change_percent": 1.31},
        "GOOGL": {"price": 141.25, "change": -0.75, "change_percent": -0.53},
        "MSFT": {"price": 378.90, "change": 4.15, "change_percent": 1.11},
        "TSLA": {"price": 245.60, "change": -3.20, "change_percent": -1.29},
    }

    data = stock_data.get(symbol.upper(), {"price": 100.00, "change": 0.00, "change_percent": 0.00})

    return {
        "symbol": symbol.upper(),
        "price": data["price"],
        "change": data["change"],
        "change_percent": data["change_percent"],
        "timestamp": datetime.now().isoformat(),
    }


def send_email(to: str, subject: str, body: str) -> dict:
    """Simulate sending an email."""
    # In a real app, this would use an email service
    return {
        "status": "sent",
        "message_id": f"msg_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "to": to,
        "subject": subject,
        "timestamp": datetime.now().isoformat(),
    }


# Map function names to their implementations
FUNCTION_MAP = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "send_email": send_email,
}


def execute_function(function_name: str, arguments: dict) -> str:
    """
    Execute a function by name with given arguments.

    Args:
        function_name: Name of the function to execute.
        arguments: Dictionary of function arguments.

    Returns:
        JSON string of the function result.
    """
    if function_name in FUNCTION_MAP:
        result = FUNCTION_MAP[function_name](**arguments)
        return json.dumps(result)
    else:
        return json.dumps({"error": f"Unknown function: {function_name}"})


def chat_with_functions(user_message: str, conversation_history: list = None) -> str:
    """
    Complete a chat with function calling support.

    Args:
        user_message: The user's message.
        conversation_history: Optional existing conversation history.

    Returns:
        The assistant's final response.
    """
    if conversation_history is None:
        conversation_history = []

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message})

    # Make the initial API call
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=conversation_history,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    # Check if the model wants to call functions
    while assistant_message.tool_calls:
        # Add assistant's message with tool calls to history
        conversation_history.append(assistant_message)

        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            console.print(f"  [cyan]Executing:[/cyan] {function_name}({arguments})")

            # Execute the function
            result = execute_function(function_name, arguments)

            console.print(f"  [green]Result:[/green] {result}")

            # Add function result to conversation history
            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # Make another API call with the function results
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=conversation_history,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

    # Add final response to history
    conversation_history.append(assistant_message)

    return assistant_message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Function Execution Example[/bold blue]\n"
            "Execute function calls and return results to the model",
            border_style="blue",
        )
    )

    # Example 1: Simple weather query
    console.print("\n[bold yellow]Example 1: Weather Query[/bold yellow]")
    console.print("[bold green]User:[/bold green] What's the weather like in Tokyo?")

    response = chat_with_functions("What's the weather like in Tokyo?")
    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")

    # Example 2: Stock price query
    console.print("\n[bold yellow]Example 2: Stock Price Query[/bold yellow]")
    console.print("[bold green]User:[/bold green] What's Apple's stock price?")

    response = chat_with_functions("What's Apple's stock price?")
    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")

    # Example 3: Send email action
    console.print("\n[bold yellow]Example 3: Send Email[/bold yellow]")
    console.print(
        "[bold green]User:[/bold green] Send an email to john@example.com about "
        "the meeting tomorrow at 3pm"
    )

    response = chat_with_functions(
        "Send an email to john@example.com with subject 'Meeting Reminder' "
        "about our meeting tomorrow at 3pm"
    )
    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")

    # Example 4: Multi-turn conversation with functions
    console.print("\n[bold yellow]Example 4: Multi-turn Conversation[/bold yellow]")

    history = []

    messages = [
        "What's the weather in London?",
        "And how about New York?",
        "Which city is warmer?",
    ]

    for msg in messages:
        console.print(f"\n[bold green]User:[/bold green] {msg}")
        response = chat_with_functions(msg, history)
        console.print(f"[bold cyan]Assistant:[/bold cyan] {response}")

    # Show the conversation structure
    console.print("\n[bold yellow]Conversation History Structure:[/bold yellow]")
    console.print(f"  Total messages: {len(history)}")

    for i, msg in enumerate(history):
        # Handle both dict and object types
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            tool_call_id = msg.get("tool_call_id", "N/A")
        else:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "") or ""
            tool_call_id = getattr(msg, "tool_call_id", "N/A")

        if role == "tool":
            console.print(f"  [{i}] {role}: (tool_call_id: {str(tool_call_id)[:20]}...)")
        elif not isinstance(msg, dict) and hasattr(msg, "tool_calls") and msg.tool_calls:
            console.print(f"  [{i}] assistant: (with {len(msg.tool_calls)} tool calls)")
        else:
            if content and len(content) > 50:
                content = content[:50] + "..."
            console.print(f"  [{i}] {role}: {content}")


if __name__ == "__main__":
    main()
