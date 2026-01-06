#!/usr/bin/env python3
"""
03_parallel_functions.py - Handling Parallel Function Calls

This example demonstrates how to handle scenarios where the model
requests multiple function calls simultaneously. This is common when
the user asks for information that requires multiple data sources.

Key concepts:
- Detecting multiple tool calls in a single response
- Executing functions in parallel
- Returning multiple results to the model
- Aggregating information from multiple sources
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()

# Tool definitions for parallel execution scenarios
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_price",
            "description": "Get flight price between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin city"},
                    "destination": {
                        "type": "string",
                        "description": "Destination city",
                    },
                    "date": {
                        "type": "string",
                        "description": "Travel date (YYYY-MM-DD)",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hotel_price",
            "description": "Get hotel prices in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "check_in": {
                        "type": "string",
                        "description": "Check-in date (YYYY-MM-DD)",
                    },
                    "nights": {"type": "integer", "description": "Number of nights"},
                },
                "required": ["city", "check_in", "nights"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get currency exchange rate",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency code",
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target currency code",
                    },
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attraction_info",
            "description": "Get information about tourist attractions",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "category": {
                        "type": "string",
                        "enum": ["museums", "landmarks", "restaurants", "parks"],
                        "description": "Type of attraction",
                    },
                },
                "required": ["city"],
            },
        },
    },
]


# Simulated function implementations
def get_weather(city: str) -> dict:
    """Simulate weather API call."""
    import time

    time.sleep(0.1)  # Simulate API latency
    weather_data = {
        "Tokyo": {"temp": 22, "condition": "Sunny", "humidity": 60},
        "Paris": {"temp": 18, "condition": "Cloudy", "humidity": 70},
        "New York": {"temp": 15, "condition": "Rainy", "humidity": 80},
        "London": {"temp": 12, "condition": "Foggy", "humidity": 85},
        "Sydney": {"temp": 28, "condition": "Clear", "humidity": 50},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown", "humidity": 50})
    return {"city": city, **data}


def get_flight_price(origin: str, destination: str, date: str) -> dict:
    """Simulate flight price API call."""
    import time

    time.sleep(0.2)  # Simulate API latency
    # Generate a pseudo-random price based on cities
    base_price = (len(origin) + len(destination)) * 50
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "price": base_price + 200,
        "currency": "USD",
        "airline": "Sample Air",
    }


def get_hotel_price(city: str, check_in: str, nights: int) -> dict:
    """Simulate hotel price API call."""
    import time

    time.sleep(0.15)  # Simulate API latency
    base_price = len(city) * 20
    return {
        "city": city,
        "check_in": check_in,
        "nights": nights,
        "price_per_night": base_price + 80,
        "total": (base_price + 80) * nights,
        "currency": "USD",
        "hotel": "Sample Hotel",
    }


def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """Simulate exchange rate API call."""
    import time

    time.sleep(0.1)  # Simulate API latency
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("EUR", "USD"): 1.09,
        ("GBP", "USD"): 1.27,
        ("JPY", "USD"): 0.0067,
    }
    rate = rates.get((from_currency, to_currency), 1.0)
    return {
        "from": from_currency,
        "to": to_currency,
        "rate": rate,
        "timestamp": datetime.now().isoformat(),
    }


def get_attraction_info(city: str, category: str = "landmarks") -> dict:
    """Simulate attraction info API call."""
    import time

    time.sleep(0.1)  # Simulate API latency
    attractions = {
        "Tokyo": {
            "landmarks": ["Tokyo Tower", "Senso-ji Temple", "Imperial Palace"],
            "museums": ["Tokyo National Museum", "teamLab Borderless"],
            "restaurants": ["Sukiyabashi Jiro", "Tsukiji Market"],
            "parks": ["Ueno Park", "Shinjuku Gyoen"],
        },
        "Paris": {
            "landmarks": ["Eiffel Tower", "Arc de Triomphe", "Notre-Dame"],
            "museums": ["Louvre", "Musee d'Orsay"],
            "restaurants": ["Le Jules Verne", "L'Ambroisie"],
            "parks": ["Luxembourg Gardens", "Tuileries Garden"],
        },
    }
    city_data = attractions.get(city, {"landmarks": ["Various attractions"]})
    return {
        "city": city,
        "category": category,
        "attractions": city_data.get(category, ["Various attractions"]),
    }


FUNCTION_MAP = {
    "get_weather": get_weather,
    "get_flight_price": get_flight_price,
    "get_hotel_price": get_hotel_price,
    "get_exchange_rate": get_exchange_rate,
    "get_attraction_info": get_attraction_info,
}


def execute_functions_sequentially(tool_calls: list) -> list:
    """Execute multiple function calls sequentially."""
    results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name in FUNCTION_MAP:
            result = FUNCTION_MAP[function_name](**arguments)
        else:
            result = {"error": f"Unknown function: {function_name}"}

        results.append(
            {
                "tool_call_id": tool_call.id,
                "result": json.dumps(result),
            }
        )
    return results


def execute_functions_parallel(tool_calls: list) -> list:
    """Execute multiple function calls in parallel using ThreadPoolExecutor."""

    def execute_single(tool_call):
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name in FUNCTION_MAP:
            result = FUNCTION_MAP[function_name](**arguments)
        else:
            result = {"error": f"Unknown function: {function_name}"}

        return {
            "tool_call_id": tool_call.id,
            "result": json.dumps(result),
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(execute_single, tool_calls))

    return results


def chat_with_parallel_functions(user_message: str, use_parallel: bool = True) -> tuple[str, dict]:
    """
    Handle a chat that may require parallel function calls.

    Args:
        user_message: The user's message.
        use_parallel: Whether to execute functions in parallel.

    Returns:
        Tuple of (response text, execution stats).
    """
    import time

    messages = [{"role": "user", "content": user_message}]
    stats = {"function_calls": 0, "execution_time": 0}

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    while assistant_message.tool_calls:
        messages.append(assistant_message)

        num_calls = len(assistant_message.tool_calls)
        stats["function_calls"] += num_calls

        console.print(f"  [cyan]Model requested {num_calls} function call(s)[/cyan]")

        # Display what functions are being called
        for tc in assistant_message.tool_calls:
            args = json.loads(tc.function.arguments)
            console.print(f"    - {tc.function.name}({args})")

        # Execute functions
        start_time = time.time()
        if use_parallel and num_calls > 1:
            console.print("  [green]Executing in parallel...[/green]")
            results = execute_functions_parallel(assistant_message.tool_calls)
        else:
            console.print("  [yellow]Executing sequentially...[/yellow]")
            results = execute_functions_sequentially(assistant_message.tool_calls)
        execution_time = time.time() - start_time
        stats["execution_time"] += execution_time

        console.print(f"  [dim]Execution time: {execution_time:.3f}s[/dim]")

        # Add results to messages
        for result in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["result"],
                }
            )

        # Continue the conversation
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

    return assistant_message.content, stats


def main():
    console.print(
        Panel.fit(
            "[bold blue]Parallel Function Calls Example[/bold blue]\n"
            "Handle multiple simultaneous function requests",
            border_style="blue",
        )
    )

    # Example 1: Trip planning (multiple parallel calls expected)
    console.print("\n[bold yellow]Example 1: Trip Planning Query[/bold yellow]")
    console.print(
        "[bold green]User:[/bold green] I'm planning a trip from New York to Tokyo "
        "on 2025-03-15 for 5 nights. What's the weather like there, "
        "how much would flights and hotels cost, and what's the USD to JPY exchange rate?"
    )

    response, stats = chat_with_parallel_functions(
        "I'm planning a trip from New York to Tokyo on 2025-03-15 for 5 nights. "
        "What's the weather like there, how much would flights and hotels cost, "
        "and what's the USD to JPY exchange rate?"
    )

    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")
    console.print(
        f"\n[dim]Stats: {stats['function_calls']} function calls, "
        f"{stats['execution_time']:.3f}s total execution time[/dim]"
    )

    # Example 2: Weather comparison (same function called multiple times)
    console.print("\n[bold yellow]Example 2: Multi-City Weather Comparison[/bold yellow]")
    console.print(
        "[bold green]User:[/bold green] Compare the weather in Tokyo, Paris, "
        "London, and Sydney right now."
    )

    response, stats = chat_with_parallel_functions(
        "Compare the weather in Tokyo, Paris, London, and Sydney right now."
    )

    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")
    console.print(
        f"\n[dim]Stats: {stats['function_calls']} function calls, "
        f"{stats['execution_time']:.3f}s total execution time[/dim]"
    )

    # Example 3: Compare sequential vs parallel execution
    console.print("\n[bold yellow]Example 3: Sequential vs Parallel Comparison[/bold yellow]")

    query = "What's the weather in Tokyo and Paris, and the flight prices between them?"

    console.print(f"[bold green]User:[/bold green] {query}")

    # Sequential execution
    console.print("\n[bold magenta]Sequential Execution:[/bold magenta]")
    _, seq_stats = chat_with_parallel_functions(query, use_parallel=False)

    # Parallel execution
    console.print("\n[bold magenta]Parallel Execution:[/bold magenta]")
    response, par_stats = chat_with_parallel_functions(query, use_parallel=True)

    console.print(f"\n[bold cyan]Response:[/bold cyan] {response}")

    # Comparison table
    table = Table(title="Execution Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Function Calls", style="green")
    table.add_column("Execution Time", style="yellow")

    table.add_row(
        "Sequential",
        str(seq_stats["function_calls"]),
        f"{seq_stats['execution_time']:.3f}s",
    )
    table.add_row(
        "Parallel",
        str(par_stats["function_calls"]),
        f"{par_stats['execution_time']:.3f}s",
    )

    if seq_stats["execution_time"] > 0:
        speedup = seq_stats["execution_time"] / max(par_stats["execution_time"], 0.001)
        table.add_row("Speedup", "-", f"{speedup:.2f}x")

    console.print(table)


if __name__ == "__main__":
    main()
