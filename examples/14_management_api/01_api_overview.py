#!/usr/bin/env python3
"""
01_api_overview.py - Management API Overview

This example provides an overview of the xAI Management API, which
allows you to manage your account, API keys, and usage programmatically.

Key concepts:
- API key information
- Account management
- Available endpoints
- Authentication
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


def get_api_key_info() -> dict:
    """
    Get information about the current API key.

    Returns:
        Dictionary containing API key metadata.
    """
    url = f"{BASE_URL}/api-key"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def list_models() -> dict:
    """
    List all available models.

    Returns:
        Dictionary containing model list.
    """
    url = f"{BASE_URL}/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def get_model_info(model_id: str) -> dict:
    """
    Get information about a specific model.

    Args:
        model_id: The model identifier.

    Returns:
        Dictionary containing model details.
    """
    url = f"{BASE_URL}/models/{model_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


def main():
    console.print(
        Panel.fit(
            "[bold blue]Management API Overview[/bold blue]\nManage your xAI account and resources",
            border_style="blue",
        )
    )

    # Check for API key
    if not API_KEY:
        console.print("[red]Error:[/red] X_AI_API_KEY environment variable not set.")
        return

    # Management API overview
    console.print("\n[bold yellow]Management API Capabilities:[/bold yellow]")
    console.print(
        """
  The Management API allows you to:

  - [cyan]API Key Management:[/cyan] View key info, permissions, usage
  - [cyan]Model Discovery:[/cyan] List available models and their capabilities
  - [cyan]Usage Tracking:[/cyan] Monitor consumption and costs
  - [cyan]Resource Management:[/cyan] Manage files, collections, etc.
"""
    )

    # Available endpoints
    console.print("\n[bold yellow]Available Endpoints:[/bold yellow]")

    endpoints_table = Table(show_header=True, header_style="bold cyan")
    endpoints_table.add_column("Category", style="green")
    endpoints_table.add_column("Endpoint")
    endpoints_table.add_column("Method")
    endpoints_table.add_column("Description")

    endpoints_table.add_row("API Key", "/v1/api-key", "GET", "Get API key info")
    endpoints_table.add_row("Models", "/v1/models", "GET", "List all models")
    endpoints_table.add_row("Models", "/v1/models/{id}", "GET", "Get model details")
    endpoints_table.add_row("Language Models", "/v1/language-models", "GET", "List language models")
    endpoints_table.add_row(
        "Image Models", "/v1/image-generation-models", "GET", "List image models"
    )
    endpoints_table.add_row("Files", "/v1/files", "GET/POST", "Manage files")
    endpoints_table.add_row("Collections", "/v1/collections", "GET/POST", "Manage collections")

    console.print(endpoints_table)

    # Example 1: Get API key info
    console.print("\n[bold yellow]Example 1: API Key Information[/bold yellow]")

    key_info = get_api_key_info()

    if "error" not in key_info:
        console.print(
            Panel(
                f"API Key ID: [cyan]{key_info.get('id', 'N/A')}[/cyan]\n"
                f"Name: {key_info.get('name', 'N/A')}\n"
                f"Team ID: {key_info.get('team_id', 'N/A')}\n"
                f"Created: {key_info.get('created_at', 'N/A')}\n"
                f"Permissions: {key_info.get('permissions', 'N/A')}",
                title="API Key Details",
                border_style="green",
            )
        )
    else:
        console.print(f"[dim]Could not retrieve API key info: {key_info.get('message')}[/dim]")

    # Example 2: List models
    console.print("\n[bold yellow]Example 2: Available Models[/bold yellow]")

    models = list_models()

    if "error" not in models:
        data = models.get("data", [])
        if data:
            model_table = Table(show_header=True, header_style="bold cyan")
            model_table.add_column("Model ID", style="green")
            model_table.add_column("Type")
            model_table.add_column("Owner")

            for model in data[:10]:
                model_table.add_row(
                    model.get("id", "N/A"),
                    model.get("object", "N/A"),
                    model.get("owned_by", "N/A"),
                )

            console.print(model_table)
            console.print(f"[dim]Total models: {len(data)}[/dim]")
        else:
            console.print("[dim]No models found.[/dim]")
    else:
        console.print(f"[dim]Could not list models: {models.get('message')}[/dim]")

    # Example 3: Get specific model info
    console.print("\n[bold yellow]Example 3: Model Details[/bold yellow]")

    model_info = get_model_info("grok-4-1-fast-reasoning")

    if "error" not in model_info:
        console.print(
            Panel(
                f"Model ID: [cyan]{model_info.get('id', 'N/A')}[/cyan]\n"
                f"Object: {model_info.get('object', 'N/A')}\n"
                f"Owner: {model_info.get('owned_by', 'N/A')}\n"
                f"Created: {model_info.get('created', 'N/A')}",
                title="Model: grok-4-1-fast-reasoning",
                border_style="cyan",
            )
        )
    else:
        console.print(f"[dim]Could not get model info: {model_info.get('message')}[/dim]")

    # Authentication
    console.print("\n[bold yellow]Authentication:[/bold yellow]")
    console.print(
        """
  All Management API requests require authentication using your API key:

  [cyan]Headers:[/cyan]
  Authorization: Bearer YOUR_API_KEY

  [cyan]Example (curl):[/cyan]
  [dim]curl https://api.x.ai/v1/api-key \\
    -H "Authorization: Bearer $X_AI_API_KEY"[/dim]

  [cyan]Example (Python):[/cyan]
  [dim]headers = {"Authorization": f"Bearer {api_key}"}
  response = requests.get("https://api.x.ai/v1/api-key", headers=headers)[/dim]
"""
    )

    # Best practices
    console.print("\n[bold yellow]Best Practices:[/bold yellow]")
    console.print(
        """
  [cyan]1. Secure your API key:[/cyan]
     - Store in environment variables or secrets manager
     - Never commit to version control
     - Use different keys for dev/prod

  [cyan]2. Monitor usage:[/cyan]
     - Track consumption regularly
     - Set up alerts for unusual usage
     - Review costs periodically

  [cyan]3. Use appropriate models:[/cyan]
     - grok-4-1-fast-reasoning for simple tasks (cheaper)
     - grok-4 for complex reasoning
     - Match model to use case

  [cyan]4. Handle errors gracefully:[/cyan]
     - Check response status codes
     - Implement retry logic for transient errors
     - Log errors for debugging
"""
    )

    # xAI Console
    console.print("\n[bold yellow]xAI Console:[/bold yellow]")
    console.print(
        """
  For a web-based management interface, visit:
  [cyan]https://console.x.ai[/cyan]

  The console provides:
  - Usage dashboards and analytics
  - API key management
  - Team and billing settings
  - Model playground
  - Documentation access
"""
    )


if __name__ == "__main__":
    main()
