#!/usr/bin/env python3
"""
04_file_management.py - Complete File Operations Overview

This example provides a comprehensive overview of all file management
operations available in the xAI API, including upload, list, retrieve,
download, and delete operations.

Key concepts:
- Complete file lifecycle management
- Error handling for file operations
- Best practices for file management
- Storage considerations
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


class FileManager:
    """Helper class for xAI file operations."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def upload(self, file_path: str, purpose: str = "assistants") -> dict:
        """Upload a file."""
        url = f"{BASE_URL}/files"

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"purpose": purpose}
            response = requests.post(url, headers=self.headers, files=files, data=data)

        return self._handle_response(response)

    def upload_bytes(self, content: bytes, filename: str, purpose: str = "assistants") -> dict:
        """Upload content from bytes."""
        url = f"{BASE_URL}/files"

        files = {"file": (filename, content)}
        data = {"purpose": purpose}
        response = requests.post(url, headers=self.headers, files=files, data=data)

        return self._handle_response(response)

    def list(
        self,
        limit: int = 100,
        order: str = "desc",
        sort_by: str = "created_at",
    ) -> dict:
        """List all files."""
        url = f"{BASE_URL}/files"
        params = {"limit": limit, "order": order, "sort_by": sort_by}

        response = requests.get(url, headers=self.headers, params=params)
        return self._handle_response(response)

    def get(self, file_id: str) -> dict:
        """Get file metadata."""
        url = f"{BASE_URL}/files/{file_id}"
        response = requests.get(url, headers=self.headers)
        return self._handle_response(response)

    def download(self, file_id: str) -> bytes | dict:
        """Download file content."""
        url = f"{BASE_URL}/files/{file_id}/content"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.content
        return self._handle_response(response)

    def delete(self, file_id: str) -> dict:
        """Delete a file."""
        url = f"{BASE_URL}/files/{file_id}"
        response = requests.delete(url, headers=self.headers)
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> dict:
        """Handle API response."""
        if response.status_code in [200, 201, 204]:
            if response.content:
                return response.json()
            return {"success": True}
        return {
            "error": True,
            "status_code": response.status_code,
            "message": response.text,
        }


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    console.print(
        Panel.fit(
            "[bold blue]File Management Overview[/bold blue]\nComplete guide to file operations",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Initialize file manager
    fm = FileManager(API_KEY)

    # Operations overview
    console.print("\n[bold yellow]File Operations Overview:[/bold yellow]")

    ops_table = Table(show_header=True, header_style="bold cyan")
    ops_table.add_column("Operation", style="green")
    ops_table.add_column("Endpoint")
    ops_table.add_column("Method")
    ops_table.add_column("Description")

    ops_table.add_row("Upload", "/v1/files", "POST", "Upload a new file")
    ops_table.add_row("List", "/v1/files", "GET", "List all files with pagination")
    ops_table.add_row("Get Metadata", "/v1/files/{id}", "GET", "Get file details")
    ops_table.add_row("Download", "/v1/files/{id}/content", "GET", "Download file content")
    ops_table.add_row("Delete", "/v1/files/{id}", "DELETE", "Remove a file")

    console.print(ops_table)

    # Demonstrate operations
    console.print("\n[bold yellow]Live Demonstration:[/bold yellow]")

    # 1. Upload a test file
    console.print("\n[cyan]1. Uploading test file...[/cyan]")
    test_content = b"This is a test file for the xAI Files API demonstration."
    result = fm.upload_bytes(test_content, "test_file.txt")

    if result.get("error"):
        console.print(f"   [red]Upload failed:[/red] {result.get('message')}")
        return

    file_id = result.get("id")
    console.print(f"   [green]Uploaded:[/green] {file_id}")

    # 2. Get file metadata
    console.print("\n[cyan]2. Retrieving file metadata...[/cyan]")
    metadata = fm.get(file_id)

    if not metadata.get("error"):
        console.print(f"   Filename: {metadata.get('filename')}")
        console.print(f"   Size: {format_size(metadata.get('bytes', 0))}")
        console.print(f"   Purpose: {metadata.get('purpose')}")
    else:
        console.print(f"   [red]Error:[/red] {metadata.get('message')}")

    # 3. List all files
    console.print("\n[cyan]3. Listing files...[/cyan]")
    files_result = fm.list(limit=5)

    if not files_result.get("error"):
        files = files_result.get("data", [])
        console.print(f"   Found {len(files)} file(s)")
        for f in files[:3]:
            console.print(f"     - {f.get('filename')}: {f.get('id')[:20]}...")
    else:
        console.print(f"   [red]Error:[/red] {files_result.get('message')}")

    # 4. Download file content
    console.print("\n[cyan]4. Downloading file content...[/cyan]")
    content = fm.download(file_id)

    if isinstance(content, bytes):
        console.print(f"   Downloaded {len(content)} bytes")
        console.print(f"   Content: {content.decode()}")
    else:
        console.print(f"   [red]Error:[/red] {content.get('message')}")

    # 5. Delete the test file
    console.print("\n[cyan]5. Deleting test file...[/cyan]")
    delete_result = fm.delete(file_id)

    if not delete_result.get("error"):
        console.print("   [green]Deleted successfully[/green]")
    else:
        console.print(f"   [red]Error:[/red] {delete_result.get('message')}")

    # Supported formats
    console.print("\n[bold yellow]Supported File Formats:[/bold yellow]")

    formats_table = Table(show_header=True, header_style="bold cyan")
    formats_table.add_column("Category", style="green")
    formats_table.add_column("Extensions")
    formats_table.add_column("MIME Types")

    formats_table.add_row("Text", ".txt, .md", "text/plain, text/markdown")
    formats_table.add_row(
        "Code",
        ".py, .js, .ts, .java, .c, .cpp, .go, .rs",
        "text/x-python, application/javascript, etc.",
    )
    formats_table.add_row("Data", ".csv, .json, .xml", "text/csv, application/json, text/xml")
    formats_table.add_row("Documents", ".pdf", "application/pdf")
    formats_table.add_row("Other", "Any UTF-8 text", "Various text/* types")

    console.print(formats_table)

    # Limits and quotas
    console.print("\n[bold yellow]Limits and Quotas:[/bold yellow]")

    limits_table = Table(show_header=True, header_style="bold cyan")
    limits_table.add_column("Limit", style="green")
    limits_table.add_column("Value")
    limits_table.add_column("Notes")

    limits_table.add_row("Max file size", "48 MB", "Per individual file")
    limits_table.add_row("Max files", "100,000", "Global limit")
    limits_table.add_row("Max storage", "100 GB", "Total storage limit")
    limits_table.add_row("List page size", "100", "Max files per list request")

    console.print(limits_table)

    # Best practices
    console.print("\n[bold yellow]Best Practices:[/bold yellow]")
    console.print(
        """
  [cyan]1. File Organization:[/cyan]
     - Use descriptive filenames
     - Include version or date in filenames if needed
     - Delete files you no longer need

  [cyan]2. Error Handling:[/cyan]
     - Always check response status codes
     - Handle rate limiting (429 errors) with retries
     - Validate file size before upload

  [cyan]3. Performance:[/cyan]
     - Upload files once, reference by ID multiple times
     - Use pagination when listing many files
     - Consider file size impact on processing time

  [cyan]4. Security:[/cyan]
     - Don't upload sensitive data unless necessary
     - Delete files containing sensitive info after use
     - Keep your API key secure
"""
    )

    # Code template
    console.print("\n[bold yellow]Code Template:[/bold yellow]")
    console.print(
        """
[dim]class XAIFileManager:
    def __init__(self, api_key):
        self.base_url = "https://api.x.ai/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def upload(self, path):
        with open(path, "rb") as f:
            return requests.post(
                f"{self.base_url}/files",
                headers=self.headers,
                files={"file": f},
                data={"purpose": "assistants"}
            ).json()

    def list(self, limit=100):
        return requests.get(
            f"{self.base_url}/files",
            headers=self.headers,
            params={"limit": limit}
        ).json()

    def delete(self, file_id):
        return requests.delete(
            f"{self.base_url}/files/{file_id}",
            headers=self.headers
        ).json()[/dim]
"""
    )


if __name__ == "__main__":
    main()
