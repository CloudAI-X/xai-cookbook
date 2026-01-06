#!/usr/bin/env python3
"""
01_upload_file.py - File Upload Basics with xAI API

This example demonstrates how to upload files to the xAI API for use
in chat conversations. The Files API enables document-based interactions
with Grok.

Key concepts:
- Uploading files via the /v1/files endpoint
- Supported file formats
- File size limits
- Understanding file IDs for later use
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


def upload_file(file_path: str, purpose: str = "assistants") -> dict:
    """
    Upload a file to the xAI API.

    Args:
        file_path: Path to the file to upload.
        purpose: Purpose of the file (default: "assistants").

    Returns:
        Dictionary containing file metadata including the file ID.
    """
    url = f"{BASE_URL}/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {"purpose": purpose}

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def upload_file_from_bytes(content: bytes, filename: str, purpose: str = "assistants") -> dict:
    """
    Upload file content from bytes.

    Args:
        content: File content as bytes.
        filename: Name for the file.
        purpose: Purpose of the file.

    Returns:
        Dictionary containing file metadata.
    """
    url = f"{BASE_URL}/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    files = {"file": (filename, content)}
    data = {"purpose": purpose}

    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def create_sample_file() -> str:
    """Create a sample text file for demonstration."""
    sample_content = """# Sample Document for xAI Files API

## Introduction
This is a sample document to demonstrate the xAI Files API capabilities.

## Key Features
1. Document understanding
2. Multi-file analysis
3. Persistent context across conversations

## Sample Data
- Project: xAI Cookbook
- Version: 1.0
- Status: Active

## Conclusion
This document can be used to test file upload and chat functionality.
"""
    sample_path = "/tmp/sample_document.md"
    with open(sample_path, "w") as f:
        f.write(sample_content)
    return sample_path


def main():
    console.print(
        Panel.fit(
            "[bold blue]File Upload Example[/bold blue]\nUpload files to the xAI API",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Supported file formats
    console.print("\n[bold yellow]Supported File Formats:[/bold yellow]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Category", style="green")
    table.add_column("Formats")
    table.add_column("Notes")

    table.add_row("Text", ".txt, .md", "Plain text and Markdown")
    table.add_row("Code", ".py, .js, .ts, .java, etc.", "Most programming languages")
    table.add_row("Data", ".csv, .json", "Structured data formats")
    table.add_row("Documents", ".pdf", "PDF documents")
    table.add_row("Other", "UTF-8 text files", "Any text-based format")

    console.print(table)

    # File limits
    console.print("\n[bold yellow]File Limits:[/bold yellow]")
    console.print("  Maximum file size: [cyan]48 MB[/cyan] per file")
    console.print("  Purpose parameter: [cyan]assistants[/cyan] (required)")

    # Example 1: Create and upload a sample file
    console.print("\n[bold yellow]Example 1: Upload Sample File[/bold yellow]")

    sample_path = create_sample_file()
    console.print(f"[dim]Created sample file: {sample_path}[/dim]")

    console.print("\n[bold green]Uploading file...[/bold green]")
    result = upload_file(sample_path)

    if "error" not in result:
        console.print(
            Panel(
                f"[green]File uploaded successfully![/green]\n\n"
                f"File ID: [cyan]{result.get('id', 'N/A')}[/cyan]\n"
                f"Filename: {result.get('filename', 'N/A')}\n"
                f"Size: {result.get('bytes', 'N/A')} bytes\n"
                f"Created: {result.get('created_at', 'N/A')}",
                title="Upload Result",
                border_style="green",
            )
        )
    else:
        console.print(f"[red]Upload failed:[/red] {result.get('message', 'Unknown error')}")

    # Example 2: Upload from bytes
    console.print("\n[bold yellow]Example 2: Upload from Bytes[/bold yellow]")

    csv_content = b"""name,age,city
Alice,30,New York
Bob,25,San Francisco
Charlie,35,Los Angeles
"""

    console.print("[bold green]Uploading CSV from bytes...[/bold green]")
    result = upload_file_from_bytes(csv_content, "sample_data.csv")

    if "error" not in result:
        console.print(
            Panel(
                f"[green]File uploaded successfully![/green]\n\n"
                f"File ID: [cyan]{result.get('id', 'N/A')}[/cyan]\n"
                f"Filename: {result.get('filename', 'N/A')}\n"
                f"Size: {result.get('bytes', 'N/A')} bytes",
                title="Upload Result",
                border_style="green",
            )
        )
    else:
        console.print(f"[red]Upload failed:[/red] {result.get('message', 'Unknown error')}")

    # API endpoint reference
    console.print("\n[bold yellow]API Reference:[/bold yellow]")

    api_table = Table(show_header=True, header_style="bold cyan")
    api_table.add_column("Operation", style="green")
    api_table.add_column("Endpoint")
    api_table.add_column("Method")

    api_table.add_row("Upload file", "/v1/files", "POST")
    api_table.add_row("List files", "/v1/files", "GET")
    api_table.add_row("Get file metadata", "/v1/files/{file_id}", "GET")
    api_table.add_row("Get file content", "/v1/files/{file_id}/content", "GET")
    api_table.add_row("Delete file", "/v1/files/{file_id}", "DELETE")

    console.print(api_table)

    # Code example
    console.print("\n[bold yellow]Code Example (requests):[/bold yellow]")
    console.print(
        """
[dim]import requests

url = "https://api.x.ai/v1/files"
headers = {"Authorization": f"Bearer {api_key}"}

with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f)}
    data = {"purpose": "assistants"}
    response = requests.post(url, headers=headers, files=files, data=data)

file_id = response.json()["id"]
print(f"Uploaded file ID: {file_id}")[/dim]
"""
    )

    # Clean up
    if os.path.exists(sample_path):
        os.remove(sample_path)
        console.print(f"\n[dim]Cleaned up sample file: {sample_path}[/dim]")


if __name__ == "__main__":
    main()
