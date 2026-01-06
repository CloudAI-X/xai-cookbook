#!/usr/bin/env python3
"""
04_complex_schemas.py - Complex Function Schemas with Nested Objects

This example demonstrates how to define sophisticated function schemas
with nested objects, arrays, and complex validation rules.

Key concepts:
- Nested object properties
- Arrays of objects
- Optional vs required fields at multiple levels
- Complex enums and validation
- oneOf/anyOf patterns
"""

import json
import os
from datetime import datetime

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

# Complex tool definitions with nested schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create a new order with multiple items, shipping details, and payment information",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {
                        "type": "object",
                        "description": "Customer information",
                        "properties": {
                            "name": {"type": "string", "description": "Full name"},
                            "email": {
                                "type": "string",
                                "description": "Email address",
                            },
                            "phone": {
                                "type": "string",
                                "description": "Phone number",
                            },
                        },
                        "required": ["name", "email"],
                    },
                    "items": {
                        "type": "array",
                        "description": "List of items in the order",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {
                                    "type": "string",
                                    "description": "Product SKU or ID",
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Product name",
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "Number of items",
                                    "minimum": 1,
                                },
                                "price": {
                                    "type": "number",
                                    "description": "Price per unit",
                                },
                                "options": {
                                    "type": "object",
                                    "description": "Product options like size, color",
                                    "properties": {
                                        "size": {
                                            "type": "string",
                                            "enum": ["XS", "S", "M", "L", "XL", "XXL"],
                                        },
                                        "color": {"type": "string"},
                                    },
                                },
                            },
                            "required": ["product_id", "name", "quantity", "price"],
                        },
                    },
                    "shipping": {
                        "type": "object",
                        "description": "Shipping address and preferences",
                        "properties": {
                            "address": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"},
                                    "state": {"type": "string"},
                                    "postal_code": {"type": "string"},
                                    "country": {"type": "string"},
                                },
                                "required": [
                                    "street",
                                    "city",
                                    "postal_code",
                                    "country",
                                ],
                            },
                            "method": {
                                "type": "string",
                                "enum": ["standard", "express", "overnight"],
                                "description": "Shipping speed",
                            },
                            "instructions": {
                                "type": "string",
                                "description": "Special delivery instructions",
                            },
                        },
                        "required": ["address", "method"],
                    },
                    "payment": {
                        "type": "object",
                        "description": "Payment method details",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["credit_card", "paypal", "bank_transfer"],
                            },
                            "billing_same_as_shipping": {"type": "boolean"},
                        },
                        "required": ["method"],
                    },
                },
                "required": ["customer", "items", "shipping", "payment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with multiple participants and complex recurrence rules",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Meeting title",
                    },
                    "description": {
                        "type": "string",
                        "description": "Meeting description",
                    },
                    "datetime": {
                        "type": "object",
                        "description": "Meeting date and time",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format",
                            },
                            "start_time": {
                                "type": "string",
                                "description": "Start time in HH:MM format (24h)",
                            },
                            "duration_minutes": {
                                "type": "integer",
                                "description": "Duration in minutes",
                            },
                            "timezone": {
                                "type": "string",
                                "description": "Timezone, e.g., 'America/New_York'",
                            },
                        },
                        "required": ["date", "start_time", "duration_minutes"],
                    },
                    "participants": {
                        "type": "array",
                        "description": "List of meeting participants",
                        "items": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "name": {"type": "string"},
                                "role": {
                                    "type": "string",
                                    "enum": ["organizer", "required", "optional"],
                                },
                                "response_status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "accepted",
                                        "declined",
                                        "tentative",
                                    ],
                                },
                            },
                            "required": ["email", "role"],
                        },
                    },
                    "recurrence": {
                        "type": "object",
                        "description": "Recurrence rules for repeating meetings",
                        "properties": {
                            "frequency": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly", "yearly"],
                            },
                            "interval": {
                                "type": "integer",
                                "description": "Interval between occurrences",
                            },
                            "days_of_week": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "monday",
                                        "tuesday",
                                        "wednesday",
                                        "thursday",
                                        "friday",
                                        "saturday",
                                        "sunday",
                                    ],
                                },
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date for recurrence (YYYY-MM-DD)",
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of occurrences",
                            },
                        },
                        "required": ["frequency"],
                    },
                    "settings": {
                        "type": "object",
                        "description": "Meeting settings",
                        "properties": {
                            "video_enabled": {"type": "boolean"},
                            "mute_on_entry": {"type": "boolean"},
                            "waiting_room": {"type": "boolean"},
                            "recording": {
                                "type": "string",
                                "enum": ["none", "local", "cloud"],
                            },
                        },
                    },
                },
                "required": ["title", "datetime", "participants"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_report",
            "description": "Generate a complex analytical report with multiple sections",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["sales", "inventory", "customer", "financial"],
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD)",
                            },
                            "end": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD)",
                            },
                            "comparison_period": {
                                "type": "string",
                                "enum": [
                                    "previous_period",
                                    "same_period_last_year",
                                    "none",
                                ],
                            },
                        },
                        "required": ["start", "end"],
                    },
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "operator": {
                                    "type": "string",
                                    "enum": [
                                        "equals",
                                        "not_equals",
                                        "contains",
                                        "greater_than",
                                        "less_than",
                                        "in",
                                    ],
                                },
                                "value": {
                                    "description": "Filter value (string, number, or array)",
                                },
                            },
                            "required": ["field", "operator", "value"],
                        },
                    },
                    "grouping": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to group by",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "aggregation": {
                                    "type": "string",
                                    "enum": ["sum", "avg", "count", "min", "max"],
                                },
                            },
                            "required": ["name", "aggregation"],
                        },
                    },
                    "output": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "enum": ["pdf", "excel", "csv", "json"],
                            },
                            "include_charts": {"type": "boolean"},
                            "email_to": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["format"],
                    },
                },
                "required": ["report_type", "date_range", "metrics", "output"],
            },
        },
    },
]


def create_order(customer: dict, items: list, shipping: dict, payment: dict) -> dict:
    """Simulate creating an order."""
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    total = sum(item["price"] * item["quantity"] for item in items)

    shipping_costs = {"standard": 5.99, "express": 12.99, "overnight": 24.99}
    shipping_cost = shipping_costs.get(shipping["method"], 5.99)

    return {
        "order_id": order_id,
        "status": "confirmed",
        "customer": customer["name"],
        "items_count": len(items),
        "subtotal": total,
        "shipping": shipping_cost,
        "total": total + shipping_cost,
        "estimated_delivery": "3-5 business days",
    }


def schedule_meeting(
    title: str,
    datetime_info: dict = None,
    participants: list = None,
    description: str = None,
    recurrence: dict = None,
    settings: dict = None,
) -> dict:
    """Simulate scheduling a meeting."""
    # Handle the datetime parameter (renamed to avoid conflict with datetime module)
    dt_info = datetime_info or {}
    meeting_id = f"MTG-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return {
        "meeting_id": meeting_id,
        "title": title,
        "date": dt_info.get("date", "TBD"),
        "time": dt_info.get("start_time", "TBD"),
        "duration": dt_info.get("duration_minutes", 60),
        "participants_count": len(participants) if participants else 0,
        "recurring": recurrence is not None,
        "video_link": f"https://meet.example.com/{meeting_id}",
        "status": "scheduled",
    }


def create_report(
    report_type: str,
    date_range: dict,
    metrics: list,
    output: dict,
    filters: list = None,
    grouping: list = None,
) -> dict:
    """Simulate creating a report."""
    report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return {
        "report_id": report_id,
        "type": report_type,
        "period": f"{date_range['start']} to {date_range['end']}",
        "metrics_count": len(metrics),
        "filters_applied": len(filters) if filters else 0,
        "format": output["format"],
        "status": "generating",
        "estimated_completion": "2 minutes",
    }


FUNCTION_MAP = {
    "create_order": create_order,
    "schedule_meeting": schedule_meeting,
    "create_report": create_report,
}


def execute_function(function_name: str, arguments: dict) -> str:
    """Execute a function with complex arguments."""
    if function_name == "schedule_meeting" and "datetime" in arguments:
        # Rename 'datetime' key to avoid conflict with module
        arguments["datetime_info"] = arguments.pop("datetime")

    if function_name in FUNCTION_MAP:
        result = FUNCTION_MAP[function_name](**arguments)
        return json.dumps(result, indent=2)
    return json.dumps({"error": f"Unknown function: {function_name}"})


def chat_with_complex_functions(user_message: str) -> str:
    """Handle chat with complex function schemas."""
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    while assistant_message.tool_calls:
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            console.print(f"\n[cyan]Function:[/cyan] {function_name}")
            console.print("[cyan]Arguments:[/cyan]")
            syntax = Syntax(json.dumps(arguments, indent=2), "json", theme="monokai")
            console.print(syntax)

            result = execute_function(function_name, arguments)

            console.print("[green]Result:[/green]")
            console.print(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

    return assistant_message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Complex Function Schemas Example[/bold blue]\n"
            "Define sophisticated nested object schemas",
            border_style="blue",
        )
    )

    # Show schema complexity
    console.print("\n[bold yellow]Schema Complexity Overview:[/bold yellow]")
    for tool in TOOLS:
        func = tool["function"]
        schema = func["parameters"]

        def count_properties(obj, depth=0):
            count = 0
            max_depth = depth
            if obj.get("type") == "object" and "properties" in obj:
                for prop in obj["properties"].values():
                    count += 1
                    sub_count, sub_depth = count_properties(prop, depth + 1)
                    count += sub_count
                    max_depth = max(max_depth, sub_depth)
            elif obj.get("type") == "array" and "items" in obj:
                sub_count, sub_depth = count_properties(obj["items"], depth + 1)
                count += sub_count
                max_depth = max(max_depth, sub_depth)
            return count, max_depth

        prop_count, max_depth = count_properties(schema)
        console.print(
            f"  [green]{func['name']}[/green]: {prop_count} properties, {max_depth} levels deep"
        )

    # Example 1: Create a complex order
    console.print("\n[bold yellow]Example 1: Creating a Complex Order[/bold yellow]")
    console.print(
        "[bold green]User:[/bold green] Create an order for John Smith "
        "(john@example.com) with 2 blue t-shirts size L at $25 each and "
        "1 pair of jeans at $50. Ship express to 123 Main St, San Francisco, CA 94102. "
        "Pay with credit card."
    )

    response = chat_with_complex_functions(
        "Create an order for John Smith (john@example.com) with 2 blue t-shirts "
        "size L at $25 each and 1 pair of jeans at $50. Ship express to "
        "123 Main St, San Francisco, CA 94102, USA. Pay with credit card."
    )

    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")

    # Example 2: Schedule a recurring meeting
    console.print("\n[bold yellow]Example 2: Scheduling a Recurring Meeting[/bold yellow]")
    console.print(
        "[bold green]User:[/bold green] Schedule a weekly team standup every "
        "Monday, Wednesday, and Friday at 9:30 AM for 15 minutes. "
        "Invite alice@example.com (required) and bob@example.com (optional). "
        "Enable video and mute on entry."
    )

    response = chat_with_complex_functions(
        "Schedule a weekly team standup every Monday, Wednesday, and Friday at "
        "9:30 AM for 15 minutes starting from 2025-02-01. "
        "Invite alice@example.com as required and bob@example.com as optional. "
        "Enable video and mute participants on entry."
    )

    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")

    # Example 3: Generate a complex report
    console.print("\n[bold yellow]Example 3: Generating a Complex Report[/bold yellow]")
    console.print(
        "[bold green]User:[/bold green] Generate a sales report for Q4 2024 "
        "(Oct 1 to Dec 31), comparing to the same period last year. "
        "Group by region and product category. Show sum of revenue and count of orders. "
        "Filter for orders over $100. Export as Excel with charts."
    )

    response = chat_with_complex_functions(
        "Generate a sales report for Q4 2024 (October 1 to December 31 2024), "
        "comparing to the same period last year. Group by region and product category. "
        "Include sum of revenue and count of orders as metrics. "
        "Only include orders with value greater than $100. "
        "Export as Excel with charts included."
    )

    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {response}")


if __name__ == "__main__":
    main()
