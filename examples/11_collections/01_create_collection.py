#!/usr/bin/env python3
"""
01_create_collection.py - Create Knowledge Base Collections

This example demonstrates how to create and manage Collections in the xAI API.
Collections are knowledge bases that allow you to organize documents and
perform semantic search across your content.

Key concepts:
- Creating new collections
- Collection configuration
- Adding documents to collections
- Collection limits and quotas
"""

import os

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

# xAI API configuration
API_KEY = os.environ.get("X_AI_API_KEY")
BASE_URL = "https://api.x.ai/v1"


def create_collection(name: str, description: str = "") -> dict:
    """
    Create a new collection (knowledge base).

    Note: The exact API endpoint may vary. Check docs.x.ai for current format.

    Args:
        name: Name for the collection.
        description: Optional description.

    Returns:
        Dictionary containing collection metadata.
    """
    url = f"{BASE_URL}/collections"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "name": name,
        "description": description,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def list_collections() -> dict:
    """
    List all collections.

    Returns:
        Dictionary containing list of collections.
    """
    url = f"{BASE_URL}/collections"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def get_collection(collection_id: str) -> dict:
    """
    Get details of a specific collection.

    Args:
        collection_id: The ID of the collection.

    Returns:
        Dictionary containing collection details.
    """
    url = f"{BASE_URL}/collections/{collection_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def upload_file_to_xai(content: bytes, filename: str) -> dict:
    """Upload a file to xAI API."""
    url = f"{BASE_URL}/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    files = {"file": (filename, content)}
    data = {"purpose": "assistants"}

    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json() if response.status_code == 200 else {"error": response.text}


def add_file_to_collection(collection_id: str, file_id: str) -> dict:
    """
    Add an uploaded file to a collection.

    Note: This is a two-step process:
    1. Upload file to xAI API (using /v1/files)
    2. Add the file to the collection

    Args:
        collection_id: The collection ID.
        file_id: The file ID from the upload step.

    Returns:
        Dictionary containing result.
    """
    url = f"{BASE_URL}/collections/{collection_id}/files"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"file_id": file_id}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def main():
    console.print(
        Panel.fit(
            "[bold blue]Create Collection Example[/bold blue]\n"
            "Build knowledge bases for document search",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Collections overview
    console.print("\n[bold yellow]What are Collections?[/bold yellow]")
    console.print(
        """
  Collections are knowledge bases that organize your documents for
  semantic search and retrieval. They enable:

  - [cyan]Document Organization[/cyan]: Group related files together
  - [cyan]Semantic Search[/cyan]: Find relevant content by meaning
  - [cyan]Hybrid Search[/cyan]: Combine keyword and semantic search
  - [cyan]Chat Integration[/cyan]: Use collections in conversations
"""
    )

    # Example 1: Create a collection
    console.print("\n[bold yellow]Example 1: Create a Collection[/bold yellow]")

    result = create_collection(
        name="xAI Cookbook Knowledge Base",
        description="Documentation and examples for the xAI API",
    )

    if "error" not in result:
        console.print(
            Panel(
                f"[green]Collection created successfully![/green]\n\n"
                f"Collection ID: [cyan]{result.get('id', 'N/A')}[/cyan]\n"
                f"Name: {result.get('name', 'N/A')}\n"
                f"Description: {result.get('description', 'N/A')}",
                title="New Collection",
                border_style="green",
            )
        )
        collection_id = result.get("id")
    else:
        console.print(
            f"[yellow]Note:[/yellow] Collection creation returned: {result.get('message')}\n"
            "[dim]The Collections API may require specific access. Check docs.x.ai[/dim]"
        )
        collection_id = None

    # Example 2: List collections
    console.print("\n[bold yellow]Example 2: List Collections[/bold yellow]")

    collections = list_collections()

    if "error" not in collections:
        data = collections.get("data", [])
        if data:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Collection ID", style="green", max_width=30)
            table.add_column("Name")
            table.add_column("Documents")

            for coll in data[:5]:
                table.add_row(
                    str(coll.get("id", "N/A"))[:30],
                    coll.get("name", "N/A"),
                    str(coll.get("document_count", 0)),
                )
            console.print(table)
        else:
            console.print("[dim]No collections found.[/dim]")
    else:
        console.print(f"[dim]Could not list collections: {collections.get('message')}[/dim]")

    # Collection workflow
    console.print("\n[bold yellow]Collection Workflow:[/bold yellow]")

    workflow_table = Table(show_header=True, header_style="bold cyan")
    workflow_table.add_column("Step", style="green")
    workflow_table.add_column("Action")
    workflow_table.add_column("Endpoint")

    workflow_table.add_row("1", "Create collection", "POST /v1/collections")
    workflow_table.add_row("2", "Upload file to xAI", "POST /v1/files")
    workflow_table.add_row("3", "Add file to collection", "POST /v1/collections/{id}/files")
    workflow_table.add_row("4", "Search collection", "POST /v1/responses with file_search")
    workflow_table.add_row("5", "Chat with collection", "Use collection_ids in chat")

    console.print(workflow_table)

    # Limits and quotas
    console.print("\n[bold yellow]Collections Limits:[/bold yellow]")

    limits_table = Table(show_header=True, header_style="bold cyan")
    limits_table.add_column("Limit", style="green")
    limits_table.add_column("Value")
    limits_table.add_column("Notes")

    limits_table.add_row("Max file size", "100 MB", "Per individual file")
    limits_table.add_row("Max files", "100,000", "Global limit")
    limits_table.add_row("Max storage", "100 GB", "Total across all collections")
    limits_table.add_row("Supported formats", "100+ types", "UTF-8 text, PDF, Word, etc.")

    console.print(limits_table)

    # Code example
    console.print("\n[bold yellow]Code Example:[/bold yellow]")
    console.print(
        """
[dim]import requests

# Step 1: Create a collection
collection = requests.post(
    "https://api.x.ai/v1/collections",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"name": "My Knowledge Base", "description": "Project docs"}
).json()

collection_id = collection["id"]

# Step 2: Upload a file
with open("document.pdf", "rb") as f:
    file = requests.post(
        "https://api.x.ai/v1/files",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": f},
        data={"purpose": "assistants"}
    ).json()

# Step 3: Add file to collection
requests.post(
    f"https://api.x.ai/v1/collections/{collection_id}/files",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"file_id": file["id"]}
)[/dim]
"""
    )

    # Important notes
    console.print("\n[bold yellow]Important Notes:[/bold yellow]")
    console.print(
        """
  - Collections require credits in your account to upload files
  - Files must be uploaded to xAI first, then added to collections
  - Semantic search uses embeddings for meaning-based retrieval
  - Hybrid search combines keyword and semantic approaches
  - Check docs.x.ai for the latest API format and endpoints
"""
    )


if __name__ == "__main__":
    main()
