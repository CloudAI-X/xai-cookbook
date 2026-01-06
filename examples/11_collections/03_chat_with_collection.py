#!/usr/bin/env python3
"""
03_chat_with_collection.py - Use Collections in Chat Conversations

This example demonstrates how to integrate Collections into chat
conversations, allowing Grok to answer questions using your knowledge base.

Key concepts:
- Attaching collections to chat
- Multi-turn conversations with collections
- Combining collections with other tools
- RAG (Retrieval Augmented Generation) pattern
"""

import os

import requests
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# OpenAI client for chat
client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()

# Direct API access
API_KEY = os.environ.get("X_AI_API_KEY")
BASE_URL = "https://api.x.ai/v1"


def chat_with_collection(
    collection_ids: list[str],
    question: str,
    system_prompt: str | None = None,
) -> dict:
    """
    Chat with Grok using collection context.

    This uses the Responses API with the collections_search tool
    to enable RAG-style conversations.

    Args:
        collection_ids: List of collection IDs to use.
        question: User's question.
        system_prompt: Optional system instructions.

    Returns:
        Dictionary containing the response.
    """
    url = f"{BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": question})

    data = {
        "model": "grok-4-1-fast-reasoning",  # Use grok-4 or grok-4-fast for best results
        "input": input_messages,
        "tools": [
            {
                "type": "file_search",
                "collection_ids": collection_ids,
            }
        ],
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def multi_turn_collection_chat(
    collection_ids: list[str],
    conversation: list[dict],
) -> dict:
    """
    Continue a multi-turn conversation with collection context.

    Args:
        collection_ids: List of collection IDs.
        conversation: List of message dictionaries.

    Returns:
        Dictionary containing the response.
    """
    url = f"{BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "grok-4-1-fast-reasoning",
        "input": conversation,
        "tools": [
            {
                "type": "file_search",
                "collection_ids": collection_ids,
            }
        ],
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


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
            "[bold blue]Chat with Collections Example[/bold blue]\n"
            "Use your knowledge base in conversations",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # RAG explanation
    console.print("\n[bold yellow]What is RAG?[/bold yellow]")
    console.print(
        """
  [cyan]RAG (Retrieval Augmented Generation)[/cyan] combines:

  1. [green]Retrieval:[/green] Search your documents for relevant content
  2. [green]Augmentation:[/green] Add retrieved content to the prompt
  3. [green]Generation:[/green] Generate response using the context

  This allows Grok to answer questions using your specific data!
"""
    )

    # Check for collections
    console.print("\n[bold yellow]Available Collections:[/bold yellow]")

    collections = list_collections()
    if collections:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("ID", max_width=30)
        table.add_column("Documents")

        for coll in collections[:5]:
            table.add_row(
                coll.get("name", "N/A"),
                str(coll.get("id", "N/A"))[:30],
                str(coll.get("document_count", 0)),
            )
        console.print(table)
        collection_ids = [collections[0].get("id")]
    else:
        console.print(
            "[yellow]No collections found.[/yellow]\n"
            "[dim]Create a collection first using 01_create_collection.py[/dim]"
        )
        collection_ids = None

    # Demonstrate chat
    if collection_ids:
        # Example 1: Basic question
        console.print("\n[bold yellow]Example 1: Basic Question[/bold yellow]")
        console.print("[bold green]Question:[/bold green] What does this collection contain?")

        result = chat_with_collection(
            collection_ids,
            "Summarize the main topics covered in these documents.",
        )

        if "error" not in result:
            console.print("\n[bold cyan]Response:[/bold cyan]")
            # Extract content from response
            output = result.get("output", [])
            for item in output:
                if item.get("type") == "message":
                    content = item.get("content", [])
                    for c in content:
                        if c.get("type") == "text":
                            console.print(Panel(c.get("text", ""), border_style="cyan"))
        else:
            console.print(f"[dim]Chat returned: {result.get('message')}[/dim]")

        # Example 2: With system prompt
        console.print("\n[bold yellow]Example 2: With System Instructions[/bold yellow]")
        console.print("[bold green]Question:[/bold green] Technical details")

        result = chat_with_collection(
            collection_ids,
            "What are the technical specifications or requirements?",
            system_prompt="You are a technical documentation assistant. Provide detailed, accurate answers based only on the documents.",
        )

        if "error" not in result:
            console.print("\n[bold cyan]Response:[/bold cyan]")
            output = result.get("output", [])
            for item in output:
                if item.get("type") == "message":
                    content = item.get("content", [])
                    for c in content:
                        if c.get("type") == "text":
                            console.print(Panel(c.get("text", ""), border_style="cyan"))
        else:
            console.print(f"[dim]Chat returned: {result.get('message')}[/dim]")

    # API pattern
    console.print("\n[bold yellow]API Pattern:[/bold yellow]")
    console.print(
        """
[dim]# Using Responses API with collections
response = requests.post(
    "https://api.x.ai/v1/responses",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "grok-4-fast",
        "input": [
            {
                "role": "system",
                "content": "Answer using only the provided documents."
            },
            {
                "role": "user",
                "content": "What are the pricing details?"
            }
        ],
        "tools": [
            {
                "type": "file_search",
                "collection_ids": ["collection-abc123", "collection-def456"]
            }
        ]
    }
)

# Parse the response
result = response.json()
for output in result.get("output", []):
    if output["type"] == "message":
        for content in output["content"]:
            if content["type"] == "text":
                print(content["text"])[/dim]
"""
    )

    # Use cases
    console.print("\n[bold yellow]Common Use Cases:[/bold yellow]")

    use_cases_table = Table(show_header=True, header_style="bold cyan")
    use_cases_table.add_column("Use Case", style="green")
    use_cases_table.add_column("Description")

    use_cases_table.add_row(
        "Documentation Q&A",
        "Answer questions about product docs",
    )
    use_cases_table.add_row(
        "Code Assistance",
        "Help with codebase understanding",
    )
    use_cases_table.add_row(
        "Knowledge Base",
        "Internal wiki or FAQ responses",
    )
    use_cases_table.add_row(
        "Research Assistant",
        "Query research papers and notes",
    )
    use_cases_table.add_row(
        "Support Bot",
        "Answer support tickets from docs",
    )

    console.print(use_cases_table)

    # Best practices
    console.print("\n[bold yellow]Best Practices:[/bold yellow]")
    console.print(
        """
  [cyan]1. Collection Organization:[/cyan]
     - Group related documents together
     - Use descriptive collection names
     - Keep collections focused on topics

  [cyan]2. Query Formulation:[/cyan]
     - Use natural language questions
     - Be specific about what you need
     - Include context when helpful

  [cyan]3. System Prompts:[/cyan]
     - Tell the model to use only document content
     - Specify response format if needed
     - Set appropriate tone and style

  [cyan]4. Multiple Collections:[/cyan]
     - Search across related collections
     - Combine different knowledge bases
     - Manage scope with collection selection
"""
    )


if __name__ == "__main__":
    main()
