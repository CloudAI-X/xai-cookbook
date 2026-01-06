#!/usr/bin/env python3
"""
04_pydantic_models.py - Using Pydantic Models for Structured Data

This example demonstrates how to use Pydantic models with xAI's structured outputs.
Pydantic provides type validation, automatic schema generation, and Python object access.

Key concepts:
- Defining Pydantic models for response structures
- Converting Pydantic models to JSON schemas
- Parsing responses into Pydantic objects
- Nested models and validators
"""

import json
import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
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


# Define Pydantic models for structured outputs


class Priority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Address(BaseModel):
    """Physical address model."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str | None = Field(default=None, description="State or province")
    postal_code: str = Field(description="ZIP or postal code")
    country: str = Field(description="Country name")


class Person(BaseModel):
    """Person information model."""

    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    address: Address | None = Field(default=None, description="Physical address")


class Task(BaseModel):
    """Task model with priority and assignment."""

    id: int = Field(description="Unique task identifier")
    title: str = Field(description="Task title")
    description: str = Field(description="Detailed description")
    priority: Priority = Field(description="Task priority level")
    assignee: str = Field(description="Person assigned to the task")
    due_date: str = Field(description="Due date in YYYY-MM-DD format")
    completed: bool = Field(default=False, description="Whether task is complete")
    estimated_hours: float = Field(description="Estimated hours to complete")


class Project(BaseModel):
    """Project model containing multiple tasks."""

    name: str = Field(description="Project name")
    description: str = Field(description="Project description")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    team_lead: Person = Field(description="Project team lead")
    tasks: list[Task] = Field(description="List of project tasks")


class MovieReview(BaseModel):
    """Movie review with sentiment analysis."""

    movie_title: str = Field(description="Title of the movie")
    reviewer: str = Field(description="Name of the reviewer")
    rating: float = Field(ge=0, le=10, description="Rating from 0 to 10")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief summary of the review")
    recommended: bool = Field(description="Whether the reviewer recommends the movie")


def get_structured_response(prompt: str, model: type[BaseModel]) -> BaseModel:
    """
    Get a structured response using a Pydantic model.

    Args:
        prompt: The user's message to send to the model.
        model: Pydantic model class defining the response structure.

    Returns:
        Parsed response as a Pydantic model instance.
    """
    # Generate JSON schema from Pydantic model
    schema = model.model_json_schema()

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a data extraction assistant. Output data exactly "
                "matching the required schema. Be precise with all fields and types.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "strict": True,
                "schema": schema,
            },
        },
    )

    content = response.choices[0].message.content
    return model.model_validate_json(content)


def main():
    console.print(
        Panel.fit(
            "[bold blue]Pydantic Models for Structured Outputs[/bold blue]\n"
            "Type-safe structured data with Pydantic validation",
            border_style="blue",
        )
    )

    # Example 1: Simple person extraction
    console.print("\n[bold yellow]Example 1: Person Extraction[/bold yellow]")

    person_schema = Person.model_json_schema()
    console.print("[dim]Generated Pydantic Schema:[/dim]")
    syntax_schema = Syntax(json.dumps(person_schema, indent=2), "json", theme="monokai")
    console.print(syntax_schema)

    person = get_structured_response(
        "Extract person info: John Smith is a software engineer who can be reached at "
        "john.smith@techcorp.com or (555) 123-4567. He lives at 123 Oak Street, "
        "San Francisco, CA 94102, USA.",
        Person,
    )

    console.print("\n[bold cyan]Extracted Person:[/bold cyan]")
    console.print(f"  Name: {person.name}")
    console.print(f"  Email: {person.email}")
    console.print(f"  Phone: {person.phone}")
    if person.address:
        console.print(f"  Address: {person.address.street}, {person.address.city}")
        console.print(f"           {person.address.state} {person.address.postal_code}")
        console.print(f"           {person.address.country}")

    # Example 2: Movie review analysis
    console.print("\n[bold yellow]Example 2: Movie Review Analysis[/bold yellow]")

    review = get_structured_response(
        "Analyze this movie review: 'The new sci-fi blockbuster Stellar Horizons is a "
        "visual masterpiece with stunning special effects and an incredible soundtrack. "
        "However, the plot is predictable and the character development feels rushed. "
        "Director Jane Doe delivers breathtaking action sequences but struggles with "
        "emotional depth. I'd give it a 7.5/10 - worth watching on the big screen for "
        "the visuals alone, but don't expect a deep story.' Reviewer: Mike Johnson",
        MovieReview,
    )

    console.print("[bold cyan]Movie Review Analysis:[/bold cyan]")
    console.print(f"  Movie: {review.movie_title}")
    console.print(f"  Reviewer: {review.reviewer}")
    console.print(f"  Rating: {review.rating}/10")
    console.print(f"  Sentiment: {review.sentiment}")
    console.print(f"  Recommended: {'Yes' if review.recommended else 'No'}")
    console.print("\n  [green]Pros:[/green]")
    for pro in review.pros:
        console.print(f"    + {pro}")
    console.print("\n  [red]Cons:[/red]")
    for con in review.cons:
        console.print(f"    - {con}")
    console.print(f"\n  [dim]Summary: {review.summary}[/dim]")

    # Example 3: Complex project with tasks
    console.print("\n[bold yellow]Example 3: Project with Tasks[/bold yellow]")

    project = get_structured_response(
        "Create a project plan for 'Website Redesign' starting 2025-02-01 and ending "
        "2025-04-30. Team lead is Sarah Wilson (sarah@company.com). Include 3 tasks: "
        "1) Design Phase - Create mockups and wireframes, high priority, assigned to "
        "Tom, due 2025-02-15, 40 hours. "
        "2) Development - Implement frontend and backend, critical priority, assigned to "
        "Lisa, due 2025-03-30, 120 hours. "
        "3) Testing - QA and user testing, medium priority, assigned to Mike, due "
        "2025-04-20, 30 hours. None are completed yet.",
        Project,
    )

    console.print("[bold cyan]Project Details:[/bold cyan]")
    console.print(f"  Name: {project.name}")
    console.print(f"  Description: {project.description}")
    console.print(f"  Duration: {project.start_date} to {project.end_date}")
    console.print(f"  Team Lead: {project.team_lead.name} ({project.team_lead.email})")

    # Display tasks in a table
    console.print("\n[bold cyan]Project Tasks:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", justify="center")
    table.add_column("Title")
    table.add_column("Priority", justify="center")
    table.add_column("Assignee")
    table.add_column("Due Date")
    table.add_column("Hours", justify="right")

    for task in project.tasks:
        priority_color = {
            Priority.CRITICAL: "red",
            Priority.HIGH: "yellow",
            Priority.MEDIUM: "blue",
            Priority.LOW: "green",
        }.get(task.priority, "white")

        table.add_row(
            str(task.id),
            task.title,
            f"[{priority_color}]{task.priority.value}[/{priority_color}]",
            task.assignee,
            task.due_date,
            str(task.estimated_hours),
        )

    console.print(table)

    # Show Pydantic validation in action
    console.print("\n[bold yellow]Pydantic Benefits:[/bold yellow]")
    console.print("  - Type validation: All fields have correct types")
    console.print("  - Enum validation: Priority values are constrained")
    console.print("  - Nested objects: Address, Person, Task properly structured")
    console.print("  - Python objects: Direct attribute access (project.name)")
    console.print(f"  - Total estimated hours: {sum(t.estimated_hours for t in project.tasks)}")


if __name__ == "__main__":
    main()
