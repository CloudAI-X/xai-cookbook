#!/usr/bin/env python3
"""
04_document_search.py - Document Search Server-Side Tool

This example demonstrates how to search through uploaded documents
using xAI's document search capability. This allows you to ask questions
about content in documents you've uploaded to xAI.

Document search is useful for:
- Querying large documents or knowledge bases
- Finding specific information in uploaded files
- Asking questions about PDF, text, or other documents

Key concepts:
- Uploading documents for search
- Querying document content
- Combining document search with other sources

Note: This example shows the pattern for document search. Actual document
upload requires the files API (covered in the files examples).
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

console = Console()


def get_client() -> OpenAI:
    """Initialize and return the xAI client."""
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] X_AI_API_KEY environment variable not set.\n"
            "Please set it in your .env file or environment."
        )
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


def search_with_context(client: OpenAI, query: str, context: str) -> str:
    """
    Search with document context provided in the message.

    This demonstrates providing document content directly in the
    conversation for questions about specific content.

    Args:
        client: The OpenAI client configured for xAI.
        query: The question about the document.
        context: The document content to search.

    Returns:
        The assistant's response.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided document content.",
            },
            {
                "role": "user",
                "content": f"Document content:\n\n{context}\n\nQuestion: {query}",
            },
        ],
    )

    return response.choices[0].message.content


def demonstrate_document_search_pattern(client: OpenAI) -> str:
    """
    Demonstrate the document search pattern.

    This shows how document search would work with uploaded files.

    Args:
        client: The OpenAI client configured for xAI.

    Returns:
        The assistant's response explaining the pattern.
    """
    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",
        messages=[
            {
                "role": "user",
                "content": (
                    "Explain how document search works in AI systems. "
                    "What are the typical steps involved in uploading a document "
                    "and then querying it?"
                ),
            }
        ],
    )

    return response.choices[0].message.content


def main():
    console.print(
        Panel.fit(
            "[bold blue]Document Search Server-Side Tool[/bold blue]\n"
            "Search through uploaded documents and knowledge bases",
            border_style="blue",
        )
    )

    client = get_client()

    # Example document content (simulating an uploaded document)
    sample_document = """
    QUARTERLY FINANCIAL REPORT - Q4 2024

    Executive Summary:
    Company XYZ reported strong Q4 results with revenue of $125 million,
    representing a 15% increase year-over-year. Operating margin improved
    to 22%, up from 18% in Q4 2023.

    Key Highlights:
    - Total revenue: $125 million (up 15% YoY)
    - Operating income: $27.5 million (up 35% YoY)
    - Net income: $21.2 million
    - Earnings per share: $1.42
    - Cash position: $89 million

    Business Segments:
    1. Consumer Products: $75 million (60% of revenue)
    2. Enterprise Services: $35 million (28% of revenue)
    3. Licensing: $15 million (12% of revenue)

    Outlook:
    For Q1 2025, the company expects revenue between $115-120 million
    with continued margin expansion. Full-year 2025 guidance will be
    provided at the annual investor meeting in March.
    """

    # Example 1: Search document for specific information
    console.print("\n[bold cyan]Example 1: Search Document for Metrics[/bold cyan]")
    console.print(
        "[dim]Query: 'What was the total revenue and how did it compare to last year?'[/dim]"
    )

    query1 = "What was the total revenue and how did it compare to last year?"
    response1 = search_with_context(client, query1, sample_document)

    console.print(Panel(Markdown(response1), title="Document Search Result", border_style="green"))

    # Example 2: Extract specific data points
    console.print("\n[bold cyan]Example 2: Extract Business Segment Data[/bold cyan]")
    console.print("[dim]Query: 'Break down the revenue by business segment'[/dim]")

    query2 = "Break down the revenue by business segment"
    response2 = search_with_context(client, query2, sample_document)

    console.print(Panel(Markdown(response2), title="Document Search Result", border_style="green"))

    # Example 3: Analyze and summarize
    console.print("\n[bold cyan]Example 3: Analyze Future Outlook[/bold cyan]")
    console.print("[dim]Query: 'What is the company's outlook and guidance for next year?'[/dim]")

    query3 = "What is the company's outlook and guidance for next year?"
    response3 = search_with_context(client, query3, sample_document)

    console.print(Panel(Markdown(response3), title="Document Search Result", border_style="green"))

    # Explain the full document search workflow
    console.print("\n[bold cyan]Document Search Workflow:[/bold cyan]")
    response_workflow = demonstrate_document_search_pattern(client)
    console.print(
        Panel(
            Markdown(response_workflow),
            title="Workflow Explanation",
            border_style="yellow",
        )
    )

    # Show document search capabilities
    console.print("\n[bold yellow]Document Search Capabilities:[/bold yellow]")
    console.print(
        """
    [cyan]Supported Document Types[/cyan]
      - PDF documents
      - Text files
      - Markdown files
      - Code files

    [cyan]Search Features[/cyan]
      - Semantic search (meaning-based)
      - Keyword extraction
      - Summarization
      - Q&A over documents

    [cyan]Integration with Files API[/cyan]
      1. Upload document using files API
      2. Reference file ID in conversation
      3. Ask questions about document content

    [cyan]Best Practices[/cyan]
      - Break large documents into sections
      - Use specific questions for better results
      - Combine with web search for additional context
    """
    )


if __name__ == "__main__":
    main()
