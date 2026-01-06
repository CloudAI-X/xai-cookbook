#!/usr/bin/env python3
"""
05_complex_schemas.py - Complex Nested Structures and Arrays

This example demonstrates advanced JSON Schema patterns for complex data structures
including deeply nested objects, arrays of arrays, conditional schemas, and more.

Key concepts:
- Deeply nested object hierarchies
- Arrays containing objects with nested arrays
- Mixed-type structures
- Real-world complex data modeling
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


def complex_json_schema(prompt: str, schema: dict, schema_name: str) -> dict:
    """
    Get a complex structured JSON response.

    Args:
        prompt: The user's message to send to the model.
        schema: Complex JSON schema definition.
        schema_name: Name identifier for the schema.

    Returns:
        Parsed JSON response as a dictionary.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a data architect assistant. Generate complex, "
                "realistic data structures that exactly match the provided schema. "
                "Be thorough and detailed in populating all fields.",
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


def visualize_structure(data: dict, name: str = "root") -> Tree:
    """Create a tree visualization of nested data structure."""
    tree = Tree(f"[bold cyan]{name}[/bold cyan]")

    def add_nodes(tree_node, obj, depth=0):
        if depth > 5:  # Prevent infinite recursion
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    branch = tree_node.add(f"[yellow]{key}[/yellow] {{...}}")
                    add_nodes(branch, value, depth + 1)
                elif isinstance(value, list):
                    branch = tree_node.add(f"[green]{key}[/green] [{len(value)} items]")
                    if value and depth < 3:
                        for i, item in enumerate(value[:2]):  # Show first 2 items
                            if isinstance(item, (dict, list)):
                                item_branch = branch.add(f"[dim][{i}][/dim]")
                                add_nodes(item_branch, item, depth + 1)
                            else:
                                branch.add(f"[dim][{i}]: {item}[/dim]")
                        if len(value) > 2:
                            branch.add(f"[dim]... and {len(value) - 2} more[/dim]")
                else:
                    tree_node.add(f"[blue]{key}[/blue]: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:3]):
                if isinstance(item, (dict, list)):
                    item_branch = tree_node.add(f"[dim][{i}][/dim]")
                    add_nodes(item_branch, item, depth + 1)

    add_nodes(tree, data)
    return tree


def main():
    console.print(
        Panel.fit(
            "[bold blue]Complex Nested Structures Example[/bold blue]\n"
            "Advanced schemas with deep nesting and arrays",
            border_style="blue",
        )
    )

    # Example 1: E-commerce order with deep nesting
    console.print("\n[bold yellow]Example 1: E-commerce Order Structure[/bold yellow]")

    ecommerce_schema = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "created_at": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pending", "processing", "shipped", "delivered", "cancelled"],
            },
            "customer": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "loyalty_tier": {
                        "type": "string",
                        "enum": ["bronze", "silver", "gold", "platinum"],
                    },
                    "addresses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["shipping", "billing"],
                                },
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "state": {"type": "string"},
                                "postal_code": {"type": "string"},
                                "country": {"type": "string"},
                                "is_default": {"type": "boolean"},
                            },
                            "required": [
                                "type",
                                "street",
                                "city",
                                "postal_code",
                                "country",
                                "is_default",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "name", "email", "loyalty_tier", "addresses"],
                "additionalProperties": False,
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "name": {"type": "string"},
                                "category": {"type": "string"},
                                "brand": {"type": "string"},
                            },
                            "required": ["sku", "name", "category", "brand"],
                            "additionalProperties": False,
                        },
                        "variant": {
                            "type": "object",
                            "properties": {
                                "size": {"type": "string"},
                                "color": {"type": "string"},
                                "material": {"type": "string"},
                            },
                            "required": ["size", "color"],
                            "additionalProperties": False,
                        },
                        "quantity": {"type": "integer"},
                        "unit_price": {"type": "number"},
                        "discount": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["percentage", "fixed"],
                                },
                                "value": {"type": "number"},
                                "code": {"type": "string"},
                            },
                            "required": ["type", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["product", "variant", "quantity", "unit_price"],
                    "additionalProperties": False,
                },
            },
            "payment": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["credit_card", "paypal", "bank_transfer", "crypto"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "authorized", "captured", "refunded"],
                    },
                    "transactions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ["authorization", "capture", "refund"],
                                },
                                "amount": {"type": "number"},
                                "timestamp": {"type": "string"},
                            },
                            "required": ["id", "type", "amount", "timestamp"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["method", "status", "transactions"],
                "additionalProperties": False,
            },
            "totals": {
                "type": "object",
                "properties": {
                    "subtotal": {"type": "number"},
                    "discount_total": {"type": "number"},
                    "tax": {"type": "number"},
                    "shipping": {"type": "number"},
                    "grand_total": {"type": "number"},
                },
                "required": [
                    "subtotal",
                    "discount_total",
                    "tax",
                    "shipping",
                    "grand_total",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "order_id",
            "created_at",
            "status",
            "customer",
            "items",
            "payment",
            "totals",
        ],
        "additionalProperties": False,
    }

    result1 = complex_json_schema(
        "Generate a realistic e-commerce order for a gold tier customer named Emily Chen "
        "ordering 2 items: a Nike running shoe (size 8, black) at $120 with 10% off, "
        "and a sports water bottle (500ml, blue) at $25. Order is being shipped to "
        "San Francisco. Payment via credit card, already captured.",
        ecommerce_schema,
        "ecommerce_order",
    )

    console.print("[bold cyan]Order Structure Tree:[/bold cyan]")
    tree = visualize_structure(result1, "Order")
    console.print(tree)

    console.print("\n[bold cyan]Full JSON Response:[/bold cyan]")
    syntax1 = Syntax(json.dumps(result1, indent=2), "json", theme="monokai")
    console.print(syntax1)

    # Example 2: Organization hierarchy
    console.print("\n[bold yellow]Example 2: Organization Hierarchy[/bold yellow]")

    org_schema = {
        "type": "object",
        "properties": {
            "company": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "founded": {"type": "integer"},
                    "industry": {"type": "string"},
                },
                "required": ["name", "founded", "industry"],
                "additionalProperties": False,
            },
            "departments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "budget": {"type": "number"},
                        "head": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "title": {"type": "string"},
                                "years_at_company": {"type": "integer"},
                            },
                            "required": ["name", "title", "years_at_company"],
                            "additionalProperties": False,
                        },
                        "teams": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "focus_area": {"type": "string"},
                                    "members": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "role": {"type": "string"},
                                                "skills": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                            "required": ["name", "role", "skills"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": ["name", "focus_area", "members"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["name", "budget", "head", "teams"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["company", "departments"],
        "additionalProperties": False,
    }

    result2 = complex_json_schema(
        "Create an organization structure for TechVision Inc, a software company "
        "founded in 2015. Include 2 departments: Engineering (budget $2M) and Product "
        "(budget $800K). Engineering has 2 teams: Backend (3 members) and Frontend "
        "(2 members). Product has 1 team: UX Design (2 members). Include realistic "
        "names, titles, and skills for each person.",
        org_schema,
        "organization",
    )

    console.print("[bold cyan]Organization Hierarchy:[/bold cyan]")
    tree2 = visualize_structure(result2, "Organization")
    console.print(tree2)

    # Calculate some stats
    total_budget = sum(d["budget"] for d in result2.get("departments", []))
    total_teams = sum(len(d["teams"]) for d in result2.get("departments", []))
    total_members = sum(
        len(t["members"]) for d in result2.get("departments", []) for t in d.get("teams", [])
    )

    console.print(
        f"\n[dim]Stats: {len(result2.get('departments', []))} departments, "
        f"{total_teams} teams, {total_members} members, "
        f"${total_budget:,.0f} total budget[/dim]"
    )

    # Example 3: Course curriculum with nested lessons
    console.print("\n[bold yellow]Example 3: Course Curriculum[/bold yellow]")

    course_schema = {
        "type": "object",
        "properties": {
            "course": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "level": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "advanced"],
                    },
                    "duration_hours": {"type": "integer"},
                    "instructor": {"type": "string"},
                },
                "required": [
                    "title",
                    "description",
                    "level",
                    "duration_hours",
                    "instructor",
                ],
                "additionalProperties": False,
            },
            "modules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "title": {"type": "string"},
                        "lessons": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["video", "reading", "quiz", "project"],
                                    },
                                    "duration_minutes": {"type": "integer"},
                                    "resources": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "type": {
                                                    "type": "string",
                                                    "enum": [
                                                        "pdf",
                                                        "code",
                                                        "link",
                                                        "video",
                                                    ],
                                                },
                                                "url": {"type": "string"},
                                            },
                                            "required": ["name", "type", "url"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": [
                                    "title",
                                    "type",
                                    "duration_minutes",
                                    "resources",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["number", "title", "lessons"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["course", "modules"],
        "additionalProperties": False,
    }

    result3 = complex_json_schema(
        "Create a Python programming course curriculum. Include 2 modules: "
        "'Python Basics' with 3 lessons (intro video, variables reading, first quiz), "
        "and 'Data Structures' with 2 lessons (lists video, final project). "
        "Each lesson should have 1-2 resources. Course is intermediate level, "
        "8 hours total, taught by Dr. Sarah Chen.",
        course_schema,
        "course_curriculum",
    )

    console.print("[bold cyan]Course Structure:[/bold cyan]")
    console.print(f"  Title: {result3['course']['title']}")
    console.print(f"  Level: {result3['course']['level']}")
    console.print(f"  Duration: {result3['course']['duration_hours']} hours")
    console.print(f"  Instructor: {result3['course']['instructor']}")

    console.print("\n[bold cyan]Modules and Lessons:[/bold cyan]")
    for module in result3.get("modules", []):
        console.print(f"\n  [yellow]Module {module['number']}: {module['title']}[/yellow]")
        for lesson in module.get("lessons", []):
            type_icon = {
                "video": "[red]Video[/red]",
                "reading": "[blue]Read[/blue]",
                "quiz": "[green]Quiz[/green]",
                "project": "[magenta]Project[/magenta]",
            }.get(lesson["type"], lesson["type"])
            console.print(
                f"    - {lesson['title']} ({type_icon}, {lesson['duration_minutes']} min)"
            )
            for resource in lesson.get("resources", []):
                console.print(f"      [dim]Resource: {resource['name']} ({resource['type']})[/dim]")


if __name__ == "__main__":
    main()
