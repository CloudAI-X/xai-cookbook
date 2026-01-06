#!/usr/bin/env python3
"""
03_strict_mode.py - Strict Mode for Guaranteed Schema Compliance

This example demonstrates strict mode with JSON Schema, which provides
stronger guarantees that the output will conform exactly to your schema.

Key concepts:
- Enabling strict mode in JSON Schema
- Guaranteed schema compliance
- Handling all required fields
- Type safety in responses
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


def strict_json_schema(prompt: str, schema: dict, schema_name: str) -> dict:
    """
    Get a JSON response with strict schema enforcement.

    Args:
        prompt: The user's message to send to the model.
        schema: JSON schema definition with strict validation.
        schema_name: Name identifier for the schema.

    Returns:
        Parsed JSON response as a dictionary.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction assistant. Output data exactly "
                "matching the required schema. Be precise with types and fields.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )

    content = response.choices[0].message.content
    return json.loads(content)


def main():
    console.print(
        Panel.fit(
            "[bold blue]Strict Mode JSON Schema Example[/bold blue]\n"
            "Guaranteed schema compliance with strict validation",
            border_style="blue",
        )
    )

    # Example 1: Event extraction with strict types
    console.print("\n[bold yellow]Example 1: Event Extraction (Strict)[/bold yellow]")

    event_schema = {
        "type": "object",
        "properties": {
            "event_name": {"type": "string"},
            "date": {"type": "string"},
            "location": {
                "type": "object",
                "properties": {
                    "venue": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                },
                "required": ["venue", "city", "country"],
                "additionalProperties": False,
            },
            "attendees": {"type": "integer"},
            "is_virtual": {"type": "boolean"},
            "ticket_price": {"type": "number"},
        },
        "required": [
            "event_name",
            "date",
            "location",
            "attendees",
            "is_virtual",
            "ticket_price",
        ],
        "additionalProperties": False,
    }

    console.print("[dim]Schema (strict mode enabled):[/dim]")
    syntax_schema1 = Syntax(json.dumps(event_schema, indent=2), "json", theme="monokai")
    console.print(syntax_schema1)

    result1 = strict_json_schema(
        "Extract event details: The annual TechConnect Summit 2025 will be held on "
        "March 15, 2025 at the Moscone Center in San Francisco, USA. Expected "
        "attendance is 5000 people. It's a hybrid event with in-person tickets at $299.",
        event_schema,
        "event_extraction",
    )

    console.print("\n[bold cyan]Extracted Event:[/bold cyan]")
    syntax1 = Syntax(json.dumps(result1, indent=2), "json", theme="monokai")
    console.print(syntax1)

    # Verify strict compliance
    console.print("\n[dim]Type verification:[/dim]")
    console.print(f"  event_name is string: {isinstance(result1.get('event_name'), str)}")
    console.print(f"  attendees is int: {isinstance(result1.get('attendees'), int)}")
    console.print(f"  is_virtual is bool: {isinstance(result1.get('is_virtual'), bool)}")
    console.print(
        f"  ticket_price is number: {isinstance(result1.get('ticket_price'), (int, float))}"
    )

    # Example 2: Structured task list with enum-like constraints
    console.print("\n[bold yellow]Example 2: Task List with Priorities[/bold yellow]")

    task_schema = {
        "type": "object",
        "properties": {
            "project_name": {"type": "string"},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                        },
                        "estimated_hours": {"type": "number"},
                        "completed": {"type": "boolean"},
                        "assignee": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "title",
                        "priority",
                        "estimated_hours",
                        "completed",
                        "assignee",
                    ],
                    "additionalProperties": False,
                },
            },
            "total_estimated_hours": {"type": "number"},
        },
        "required": ["project_name", "tasks", "total_estimated_hours"],
        "additionalProperties": False,
    }

    result2 = strict_json_schema(
        "Create a task list for a website redesign project with 4 tasks: "
        "1) Design mockups (high priority, 20 hours, assigned to Sarah), "
        "2) Frontend implementation (critical, 40 hours, assigned to Mike), "
        "3) Backend API updates (medium, 15 hours, assigned to Alex), "
        "4) Testing and QA (high, 10 hours, assigned to Lisa). None are completed yet.",
        task_schema,
        "task_list",
    )

    console.print("[bold cyan]Task List:[/bold cyan]")
    syntax2 = Syntax(json.dumps(result2, indent=2), "json", theme="monokai")
    console.print(syntax2)

    # Display as table
    console.print("\n[dim]Tasks as Table:[/dim]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", justify="center")
    table.add_column("Title")
    table.add_column("Priority", justify="center")
    table.add_column("Hours", justify="right")
    table.add_column("Assignee")
    table.add_column("Done", justify="center")

    for task in result2.get("tasks", []):
        priority_color = {
            "critical": "red",
            "high": "yellow",
            "medium": "blue",
            "low": "green",
        }.get(task["priority"], "white")

        table.add_row(
            str(task["id"]),
            task["title"],
            f"[{priority_color}]{task['priority']}[/{priority_color}]",
            str(task["estimated_hours"]),
            task["assignee"],
            "[green]Yes[/green]" if task["completed"] else "[red]No[/red]",
        )

    console.print(table)
    console.print(f"\n[bold]Total Estimated Hours:[/bold] {result2.get('total_estimated_hours')}")

    # Example 3: API response structure
    console.print("\n[bold yellow]Example 3: API Response Structure[/bold yellow]")

    api_response_schema = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "status_code": {"type": "integer"},
            "message": {"type": "string"},
            "data": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "username": {"type": "string"},
                    "permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "last_login": {"type": "string"},
                },
                "required": ["user_id", "username", "permissions", "last_login"],
                "additionalProperties": False,
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "api_version": {"type": "string"},
                },
                "required": ["request_id", "timestamp", "api_version"],
                "additionalProperties": False,
            },
        },
        "required": ["success", "status_code", "message", "data", "metadata"],
        "additionalProperties": False,
    }

    result3 = strict_json_schema(
        "Generate a sample successful API response for a user authentication check. "
        "The user is admin123 with username 'administrator', permissions for read, write, "
        "and delete. Last login was 2025-01-05 at 14:30 UTC.",
        api_response_schema,
        "api_response",
    )

    console.print("[bold cyan]API Response:[/bold cyan]")
    syntax3 = Syntax(json.dumps(result3, indent=2), "json", theme="monokai")
    console.print(syntax3)

    # Verify no additional properties
    console.print("\n[dim]Strict mode verification:[/dim]")
    expected_keys = {"success", "status_code", "message", "data", "metadata"}
    actual_keys = set(result3.keys())
    console.print(f"  No extra top-level fields: {actual_keys == expected_keys}")

    expected_data_keys = {"user_id", "username", "permissions", "last_login"}
    actual_data_keys = set(result3.get("data", {}).keys())
    console.print(f"  No extra data fields: {actual_data_keys == expected_data_keys}")


if __name__ == "__main__":
    main()
