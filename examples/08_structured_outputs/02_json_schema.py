#!/usr/bin/env python3
"""
02_json_schema.py - JSON with Schema Validation

This example demonstrates how to use JSON Schema with xAI's chat completions API.
JSON Schema provides structure validation, ensuring responses match your expected format.

Key concepts:
- Defining JSON schemas for response validation
- Schema properties and types
- Required fields specification
- Array and nested object schemas
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


def json_with_schema(prompt: str, schema: dict, schema_name: str) -> dict:
    """
    Get a JSON response validated against a schema.

    Args:
        prompt: The user's message to send to the model.
        schema: JSON schema definition for validation.
        schema_name: Name identifier for the schema.

    Returns:
        Parsed JSON response as a dictionary.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that outputs structured JSON data. "
                "Follow the provided schema exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
            },
        },
    )

    content = response.choices[0].message.content
    return json.loads(content)


def main():
    console.print(
        Panel.fit(
            "[bold blue]JSON Schema Validation Example[/bold blue]\n"
            "Enforcing response structure with JSON Schema",
            border_style="blue",
        )
    )

    # Example 1: Simple list schema
    console.print("\n[bold yellow]Example 1: Languages List Schema[/bold yellow]")

    languages_schema = {
        "type": "object",
        "properties": {
            "languages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of programming language names",
            },
        },
        "required": ["languages"],
    }

    console.print("[dim]Schema:[/dim]")
    syntax_schema1 = Syntax(json.dumps(languages_schema, indent=2), "json", theme="monokai")
    console.print(syntax_schema1)

    result1 = json_with_schema(
        "List 5 popular programming languages for web development.",
        languages_schema,
        "languages_list",
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    syntax1 = Syntax(json.dumps(result1, indent=2), "json", theme="monokai")
    console.print(syntax1)

    # Example 2: Object with multiple properties
    console.print("\n[bold yellow]Example 2: User Profile Schema[/bold yellow]")

    user_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User's full name"},
            "age": {"type": "integer", "description": "User's age in years"},
            "email": {"type": "string", "description": "User's email address"},
            "active": {
                "type": "boolean",
                "description": "Whether the account is active",
            },
            "interests": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of user interests",
            },
        },
        "required": ["name", "age", "email", "active", "interests"],
    }

    console.print("[dim]Schema:[/dim]")
    syntax_schema2 = Syntax(json.dumps(user_schema, indent=2), "json", theme="monokai")
    console.print(syntax_schema2)

    result2 = json_with_schema(
        "Create a sample user profile for a software developer named Alex.",
        user_schema,
        "user_profile",
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    syntax2 = Syntax(json.dumps(result2, indent=2), "json", theme="monokai")
    console.print(syntax2)

    # Example 3: Nested objects schema
    console.print("\n[bold yellow]Example 3: Nested Object Schema[/bold yellow]")

    book_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "nationality": {"type": "string"},
                    "birth_year": {"type": "integer"},
                },
                "required": ["name"],
            },
            "publication": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "publisher": {"type": "string"},
                    "isbn": {"type": "string"},
                },
                "required": ["year", "publisher"],
            },
            "genres": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["title", "author", "publication", "genres"],
    }

    console.print("[dim]Schema:[/dim]")
    syntax_schema3 = Syntax(json.dumps(book_schema, indent=2), "json", theme="monokai")
    console.print(syntax_schema3)

    result3 = json_with_schema(
        "Provide information about the book '1984' by George Orwell.",
        book_schema,
        "book_info",
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    syntax3 = Syntax(json.dumps(result3, indent=2), "json", theme="monokai")
    console.print(syntax3)

    # Example 4: Array of objects
    console.print("\n[bold yellow]Example 4: Array of Objects Schema[/bold yellow]")

    products_schema = {
        "type": "object",
        "properties": {
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "in_stock": {"type": "boolean"},
                    },
                    "required": ["id", "name", "price", "in_stock"],
                },
            },
            "total_count": {"type": "integer"},
        },
        "required": ["products", "total_count"],
    }

    console.print("[dim]Schema:[/dim]")
    syntax_schema4 = Syntax(json.dumps(products_schema, indent=2), "json", theme="monokai")
    console.print(syntax_schema4)

    result4 = json_with_schema(
        "Create a list of 3 electronic products with their details.",
        products_schema,
        "products_list",
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    syntax4 = Syntax(json.dumps(result4, indent=2), "json", theme="monokai")
    console.print(syntax4)

    # Verify schema conformance
    console.print("\n[bold yellow]Schema Conformance Check:[/bold yellow]")
    console.print(f"  Products count: {len(result4.get('products', []))}")
    console.print(f"  Total count field: {result4.get('total_count')}")
    console.print("  All products have required fields: ", end="")

    all_valid = all(
        all(key in product for key in ["id", "name", "price", "in_stock"])
        for product in result4.get("products", [])
    )
    console.print("[green]Yes[/green]" if all_valid else "[red]No[/red]")


if __name__ == "__main__":
    main()
