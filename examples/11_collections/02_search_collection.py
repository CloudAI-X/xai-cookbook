#!/usr/bin/env python3
"""
02_search_collection.py - Search Knowledge Base Collections

This example demonstrates how to search documents within Collections
using semantic, keyword, and hybrid search modes.

Key concepts:
- Semantic search (meaning-based)
- Keyword search (exact matching)
- Hybrid search (combined)
- Search result ranking
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


def search_collection(
    collection_id: str,
    query: str,
    retrieval_mode: str = "hybrid",
    max_results: int = 10,
) -> dict:
    """
    Search documents in a collection.

    Args:
        collection_id: The collection to search.
        query: Search query text.
        retrieval_mode: "semantic", "keyword", or "hybrid" (default).
        max_results: Maximum number of results.

    Returns:
        Dictionary containing search results.
    """
    # Using the Responses API with file_search tool
    url = f"{BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "grok-4-1-fast-reasoning",
        "input": [
            {
                "role": "user",
                "content": f"Search for: {query}",
            }
        ],
        "tools": [
            {
                "type": "file_search",
                "collection_ids": [collection_id],
                "retrieval_mode": {"type": retrieval_mode},
                "max_results": max_results,
            }
        ],
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def semantic_search(collection_id: str, query: str) -> dict:
    """
    Perform semantic (meaning-based) search.

    Semantic search finds documents based on conceptual similarity,
    not just keyword matching.

    Args:
        collection_id: The collection to search.
        query: Natural language query.

    Returns:
        Search results.
    """
    return search_collection(
        collection_id=collection_id,
        query=query,
        retrieval_mode="semantic",
    )


def keyword_search(collection_id: str, query: str) -> dict:
    """
    Perform keyword (exact match) search.

    Keyword search finds documents containing specific terms.

    Args:
        collection_id: The collection to search.
        query: Keywords to search for.

    Returns:
        Search results.
    """
    return search_collection(
        collection_id=collection_id,
        query=query,
        retrieval_mode="keyword",
    )


def hybrid_search(collection_id: str, query: str) -> dict:
    """
    Perform hybrid search (semantic + keyword).

    Hybrid search combines both approaches for best results.

    Args:
        collection_id: The collection to search.
        query: Search query.

    Returns:
        Search results.
    """
    return search_collection(
        collection_id=collection_id,
        query=query,
        retrieval_mode="hybrid",
    )


def list_collections() -> list:
    """Get list of available collections."""
    url = f"{BASE_URL}/collections"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []


def main():
    console.print(
        Panel.fit(
            "[bold blue]Search Collections Example[/bold blue]\n"
            "Find documents using semantic and keyword search",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Search modes explanation
    console.print("\n[bold yellow]Search Modes:[/bold yellow]")

    modes_table = Table(show_header=True, header_style="bold cyan")
    modes_table.add_column("Mode", style="green")
    modes_table.add_column("Description")
    modes_table.add_column("Best For")

    modes_table.add_row(
        "semantic",
        "Finds conceptually similar content",
        "Natural language questions, concept search",
    )
    modes_table.add_row(
        "keyword",
        "Matches exact terms and phrases",
        "Specific terms, IDs, code snippets",
    )
    modes_table.add_row(
        "hybrid",
        "Combines semantic and keyword",
        "General search (default, recommended)",
    )

    console.print(modes_table)

    # Check for existing collections
    console.print("\n[bold yellow]Available Collections:[/bold yellow]")

    collections = list_collections()
    if collections:
        for coll in collections[:5]:
            console.print(f"  - {coll.get('name', 'N/A')}: [cyan]{coll.get('id', 'N/A')}[/cyan]")
        collection_id = collections[0].get("id")
    else:
        console.print(
            "[yellow]No collections found.[/yellow]\n"
            "[dim]Create a collection first using 01_create_collection.py[/dim]"
        )
        collection_id = None

    # Demonstrate search (with placeholder if no collection)
    if collection_id:
        # Example 1: Semantic search
        console.print("\n[bold yellow]Example 1: Semantic Search[/bold yellow]")
        console.print("[bold green]Query:[/bold green] How do I authenticate?")

        result = semantic_search(collection_id, "How do I authenticate with the API?")

        if "error" not in result:
            console.print("\n[bold cyan]Results:[/bold cyan]")
            console.print(Panel(str(result)[:500] + "...", border_style="cyan"))
        else:
            console.print(f"[dim]Search returned: {result.get('message')}[/dim]")

        # Example 2: Keyword search
        console.print("\n[bold yellow]Example 2: Keyword Search[/bold yellow]")
        console.print("[bold green]Query:[/bold green] API_KEY")

        result = keyword_search(collection_id, "API_KEY")

        if "error" not in result:
            console.print("\n[bold cyan]Results:[/bold cyan]")
            console.print(Panel(str(result)[:500] + "...", border_style="cyan"))
        else:
            console.print(f"[dim]Search returned: {result.get('message')}[/dim]")

        # Example 3: Hybrid search
        console.print("\n[bold yellow]Example 3: Hybrid Search[/bold yellow]")
        console.print("[bold green]Query:[/bold green] rate limits and quotas")

        result = hybrid_search(collection_id, "rate limits and quotas")

        if "error" not in result:
            console.print("\n[bold cyan]Results:[/bold cyan]")
            console.print(Panel(str(result)[:500] + "...", border_style="cyan"))
        else:
            console.print(f"[dim]Search returned: {result.get('message')}[/dim]")

    # API format reference
    console.print("\n[bold yellow]API Format Reference:[/bold yellow]")
    console.print(
        """
[dim]# Using the Responses API with file_search tool
response = requests.post(
    "https://api.x.ai/v1/responses",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "grok-4-fast",
        "input": [
            {"role": "user", "content": "Search query here"}
        ],
        "tools": [
            {
                "type": "file_search",
                "collection_ids": ["collection-abc123"],
                "retrieval_mode": {"type": "hybrid"},  # or "semantic", "keyword"
                "max_results": 10
            }
        ]
    }
)[/dim]
"""
    )

    # Search tips
    console.print("\n[bold yellow]Search Tips:[/bold yellow]")
    console.print(
        """
  [cyan]Semantic Search:[/cyan]
    - Use natural language questions
    - Good for "how do I..." or "what is..." queries
    - Finds conceptually related content

  [cyan]Keyword Search:[/cyan]
    - Use specific terms, names, or IDs
    - Good for finding exact matches
    - Use quotes for phrases

  [cyan]Hybrid Search (Default):[/cyan]
    - Best of both approaches
    - Recommended for general use
    - Handles diverse query types
"""
    )

    # Search parameters
    console.print("\n[bold yellow]Search Parameters:[/bold yellow]")

    params_table = Table(show_header=True, header_style="bold cyan")
    params_table.add_column("Parameter", style="green")
    params_table.add_column("Type")
    params_table.add_column("Description")

    params_table.add_row(
        "collection_ids",
        "array",
        "List of collection IDs to search",
    )
    params_table.add_row(
        "retrieval_mode",
        "object",
        '{"type": "semantic|keyword|hybrid"}',
    )
    params_table.add_row(
        "max_results",
        "integer",
        "Maximum results to return",
    )

    console.print(params_table)


if __name__ == "__main__":
    main()
