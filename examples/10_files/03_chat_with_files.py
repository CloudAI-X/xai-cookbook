#!/usr/bin/env python3
"""
03_chat_with_files.py - Chat with Uploaded Files

This example demonstrates how to reference uploaded files in chat
conversations with Grok. The model can read, analyze, and answer
questions about your documents.

Key concepts:
- Attaching files to chat messages
- Document search and analysis
- Multi-file conversations
- Persistent file context
"""

import os

import requests
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# OpenAI client for chat completions
client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()

# Direct API access for file operations
API_KEY = os.environ.get("X_AI_API_KEY")
BASE_URL = "https://api.x.ai/v1"


def upload_file(file_path: str) -> dict:
    """Upload a file and return its metadata."""
    url = f"{BASE_URL}/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {"purpose": "assistants"}
        response = requests.post(url, headers=headers, files=files, data=data)

    return response.json() if response.status_code == 200 else {"error": response.text}


def list_files() -> list:
    """List all uploaded files."""
    url = f"{BASE_URL}/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(url, headers=headers)
    return response.json().get("data", []) if response.status_code == 200 else []


def chat_with_file(file_id: str, question: str) -> str:
    """
    Send a chat message with a file attachment.

    Note: This uses the Responses API pattern for file-based chat.
    The exact API may vary - check docs.x.ai for current implementation.

    Args:
        file_id: ID of the uploaded file.
        question: Question about the file.

    Returns:
        The model's response.
    """
    # Using extra_body to pass file attachments
    # Note: The exact parameter name may be 'file_ids', 'attachments', etc.
    # Check the current xAI documentation for the exact format

    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",  # Use an agentic model like grok-4 for best results
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            # File attachment - format may vary by API version
            extra_body={
                "file_ids": [file_id],
            },
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def chat_with_multiple_files(file_ids: list[str], question: str) -> str:
    """
    Chat with multiple files attached.

    Args:
        file_ids: List of file IDs to attach.
        question: Question about the files.

    Returns:
        The model's response.
    """
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            extra_body={
                "file_ids": file_ids,
            },
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def create_sample_files() -> list[str]:
    """Create sample files for demonstration."""
    files = []

    # Sample 1: Project overview
    content1 = """# Project Alpha Overview

## Summary
Project Alpha is a machine learning pipeline for sentiment analysis.

## Key Metrics
- Accuracy: 94.5%
- F1 Score: 0.92
- Training Time: 4.5 hours

## Team
- Lead: Dr. Smith
- Engineers: 5
- Budget: $500,000

## Timeline
- Start: January 2024
- End: June 2024
"""
    path1 = "/tmp/project_alpha.md"
    with open(path1, "w") as f:
        f.write(content1)
    files.append(path1)

    # Sample 2: Data summary
    content2 = """name,department,salary,years
Alice,Engineering,95000,5
Bob,Marketing,75000,3
Charlie,Engineering,105000,7
Diana,Sales,85000,4
Eve,Engineering,90000,2
"""
    path2 = "/tmp/employees.csv"
    with open(path2, "w") as f:
        f.write(content2)
    files.append(path2)

    return files


def main():
    console.print(
        Panel.fit(
            "[bold blue]Chat with Files Example[/bold blue]\nUse uploaded files in conversations",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Show existing files
    console.print("\n[bold yellow]Current Uploaded Files:[/bold yellow]")
    existing_files = list_files()

    if existing_files:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("File ID", style="green", max_width=30)
        table.add_column("Filename")

        for file in existing_files[:5]:
            table.add_row(
                file.get("id", "N/A")[:30],
                file.get("filename", "N/A"),
            )
        console.print(table)
    else:
        console.print("[yellow]No files uploaded yet.[/yellow]")

    # Create and upload sample files
    console.print("\n[bold yellow]Creating Sample Files...[/bold yellow]")
    sample_paths = create_sample_files()

    uploaded_ids = []
    for path in sample_paths:
        console.print(f"  Uploading {os.path.basename(path)}...")
        result = upload_file(path)
        if "id" in result:
            uploaded_ids.append(result["id"])
            console.print(f"    [green]Success:[/green] {result['id'][:30]}...")
        else:
            console.print(f"    [red]Failed:[/red] {result.get('error', 'Unknown')}")

    if not uploaded_ids:
        console.print("[red]No files uploaded. Cannot proceed with chat.[/red]")
        return

    # Example 1: Chat with single file
    console.print("\n[bold yellow]Example 1: Chat with Single File[/bold yellow]")
    console.print("[bold green]Question:[/bold green] What is Project Alpha about?")

    response = chat_with_file(
        uploaded_ids[0],
        "What is this project about? Summarize the key points.",
    )

    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Panel(response, border_style="cyan"))

    # Example 2: Chat with multiple files
    if len(uploaded_ids) > 1:
        console.print("\n[bold yellow]Example 2: Chat with Multiple Files[/bold yellow]")
        console.print("[bold green]Question:[/bold green] Analyze both documents")

        response = chat_with_multiple_files(
            uploaded_ids,
            "I have a project overview and employee data. "
            "How many engineers are mentioned in the employee data? "
            "What is the project's accuracy metric?",
        )

        console.print("\n[bold cyan]Response:[/bold cyan]")
        console.print(Panel(response, border_style="cyan"))

    # Important notes
    console.print("\n[bold yellow]Important Notes:[/bold yellow]")

    notes_table = Table(show_header=True, header_style="bold cyan")
    notes_table.add_column("Feature", style="green")
    notes_table.add_column("Details")

    notes_table.add_row(
        "Model Support",
        "Use agentic models (grok-4, grok-4-fast) for best results",
    )
    notes_table.add_row(
        "Document Search",
        "Model automatically searches documents when files attached",
    )
    notes_table.add_row(
        "Multi-turn",
        "File context persists across conversation turns",
    )
    notes_table.add_row(
        "Pricing",
        "$10 per 1,000 document search tool invocations",
    )
    notes_table.add_row(
        "Max File Size",
        "48 MB per file",
    )

    console.print(notes_table)

    # API pattern reference
    console.print("\n[bold yellow]API Pattern Reference:[/bold yellow]")
    console.print(
        """
[dim]# Using the Responses API (recommended for files)
response = requests.post(
    "https://api.x.ai/v1/responses",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "grok-4-fast",
        "input": [
            {
                "role": "user",
                "content": "Summarize this document",
            }
        ],
        "tools": [
            {
                "type": "file_search",
                "file_ids": ["file-abc123"],
            }
        ],
    }
)

# Check docs.x.ai for the latest API format[/dim]
"""
    )

    # Clean up sample files
    for path in sample_paths:
        if os.path.exists(path):
            os.remove(path)

    console.print("\n[dim]Sample files cleaned up.[/dim]")


if __name__ == "__main__":
    main()
