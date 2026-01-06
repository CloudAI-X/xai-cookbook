#!/usr/bin/env python3
"""
02_list_files.py - List and Manage Uploaded Files

This example demonstrates how to list, retrieve, and manage files
that have been uploaded to the xAI API.

Key concepts:
- Listing all uploaded files
- Pagination and sorting
- Retrieving file metadata
- Downloading file content
"""

import os
from datetime import datetime

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


def list_files(
    limit: int = 10,
    order: str = "desc",
    sort_by: str = "created_at",
    pagination_token: str | None = None,
) -> dict:
    """
    List uploaded files with pagination and sorting.

    Args:
        limit: Maximum number of files to return (max 100).
        order: Sort order ("asc" or "desc").
        sort_by: Field to sort by ("created_at", "filename", "size").
        pagination_token: Token for pagination.

    Returns:
        Dictionary containing file list and pagination info.
    """
    url = f"{BASE_URL}/files"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {
        "limit": min(limit, 100),
        "order": order,
        "sort_by": sort_by,
    }

    if pagination_token:
        params["pagination_token"] = pagination_token

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def get_file_metadata(file_id: str) -> dict:
    """
    Get metadata for a specific file.

    Args:
        file_id: The ID of the file.

    Returns:
        Dictionary containing file metadata.
    """
    url = f"{BASE_URL}/files/{file_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def download_file_content(file_id: str) -> bytes | dict:
    """
    Download the content of a file.

    Args:
        file_id: The ID of the file.

    Returns:
        File content as bytes, or error dictionary.
    """
    url = f"{BASE_URL}/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.content
    else:
        return {"error": response.status_code, "message": response.text}


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_timestamp(timestamp: int | str) -> str:
    """Format timestamp to readable date."""
    if isinstance(timestamp, int):
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return str(timestamp)


def main():
    console.print(
        Panel.fit(
            "[bold blue]List Files Example[/bold blue]\nView and manage uploaded files",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Example 1: List all files
    console.print("\n[bold yellow]Example 1: List All Files[/bold yellow]")

    result = list_files(limit=20, order="desc", sort_by="created_at")

    if "error" not in result:
        files = result.get("data", [])

        if files:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("File ID", style="green", max_width=30)
            table.add_column("Filename")
            table.add_column("Size", justify="right")
            table.add_column("Created")

            for file in files[:10]:  # Show first 10
                table.add_row(
                    file.get("id", "N/A")[:30],
                    file.get("filename", "N/A"),
                    format_file_size(file.get("bytes", 0)),
                    format_timestamp(file.get("created_at", "N/A")),
                )

            console.print(table)
            console.print(f"\n[dim]Total files shown: {len(files[:10])}[/dim]")

            # Check for pagination
            if result.get("has_more"):
                console.print(
                    f"[dim]More files available. "
                    f"Use pagination_token: {result.get('next_token')}[/dim]"
                )
        else:
            console.print("[yellow]No files found.[/yellow]")
            console.print("[dim]Upload files first using 01_upload_file.py[/dim]")

        # Example 2: Get specific file metadata
        if files:
            console.print("\n[bold yellow]Example 2: Get File Metadata[/bold yellow]")

            first_file = files[0]
            file_id = first_file.get("id")

            console.print(f"[dim]Fetching metadata for: {file_id}[/dim]")

            metadata = get_file_metadata(file_id)

            if "error" not in metadata:
                console.print(
                    Panel(
                        f"File ID: [cyan]{metadata.get('id', 'N/A')}[/cyan]\n"
                        f"Filename: {metadata.get('filename', 'N/A')}\n"
                        f"Size: {format_file_size(metadata.get('bytes', 0))}\n"
                        f"Purpose: {metadata.get('purpose', 'N/A')}\n"
                        f"Created: {format_timestamp(metadata.get('created_at', 'N/A'))}\n"
                        f"Team ID: {metadata.get('team_id', 'N/A')}",
                        title="File Metadata",
                        border_style="cyan",
                    )
                )
            else:
                console.print(f"[red]Error fetching metadata:[/red] {metadata.get('message')}")

            # Example 3: Download file content
            console.print("\n[bold yellow]Example 3: Download File Content[/bold yellow]")

            content = download_file_content(file_id)

            if isinstance(content, bytes):
                preview = content[:500].decode("utf-8", errors="replace")
                console.print(
                    Panel(
                        f"{preview}{'...' if len(content) > 500 else ''}",
                        title=f"Content Preview ({len(content)} bytes)",
                        border_style="green",
                    )
                )
            else:
                console.print(f"[red]Error downloading:[/red] {content.get('message')}")

    else:
        console.print(f"[red]Error listing files:[/red] {result.get('message')}")

    # Sorting options reference
    console.print("\n[bold yellow]Listing Options:[/bold yellow]")

    options_table = Table(show_header=True, header_style="bold cyan")
    options_table.add_column("Parameter", style="green")
    options_table.add_column("Values")
    options_table.add_column("Description")

    options_table.add_row("limit", "1-100", "Max files to return")
    options_table.add_row("order", '"asc", "desc"', "Sort order")
    options_table.add_row(
        "sort_by",
        '"created_at", "filename", "size"',
        "Field to sort by",
    )
    options_table.add_row("pagination_token", "string", "Token for next page")

    console.print(options_table)

    # Code example
    console.print("\n[bold yellow]Code Example:[/bold yellow]")
    console.print(
        """
[dim]import requests

# List files
url = "https://api.x.ai/v1/files"
headers = {"Authorization": f"Bearer {api_key}"}
params = {"limit": 20, "order": "desc", "sort_by": "created_at"}

response = requests.get(url, headers=headers, params=params)
files = response.json()["data"]

for file in files:
    print(f"{file['filename']}: {file['id']}")[/dim]
"""
    )


if __name__ == "__main__":
    main()
