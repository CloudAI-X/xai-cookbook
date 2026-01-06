#!/usr/bin/env python3
"""
01_error_handling.py - Comprehensive Error Handling

This example demonstrates comprehensive error handling strategies for
the xAI API, including handling various error types, logging, and
recovery mechanisms.

Key concepts:
- API error types and status codes
- Exception handling patterns
- Error logging and monitoring
- Graceful degradation
"""

import logging
import os
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

client = OpenAI(
    api_key=os.environ.get("X_AI_API_KEY", "invalid-key"),
    base_url="https://api.x.ai/v1",
)


class ErrorCategory(Enum):
    """Categories of API errors."""

    AUTH = "authentication"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    SERVER = "server"
    NETWORK = "network"
    UNKNOWN = "unknown"


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize an exception into an error category."""
    if isinstance(error, AuthenticationError):
        return ErrorCategory.AUTH
    elif isinstance(error, RateLimitError):
        return ErrorCategory.RATE_LIMIT
    elif isinstance(error, (BadRequestError, UnprocessableEntityError)):
        return ErrorCategory.VALIDATION
    elif isinstance(error, InternalServerError):
        return ErrorCategory.SERVER
    elif isinstance(error, (APIConnectionError, APITimeoutError)):
        return ErrorCategory.NETWORK
    else:
        return ErrorCategory.UNKNOWN


def handle_api_error(error: Exception, context: str = "") -> dict:
    """
    Handle an API error with logging and structured response.

    Args:
        error: The exception that occurred.
        context: Additional context about the operation.

    Returns:
        Dictionary with error details and suggested action.
    """
    category = categorize_error(error)
    timestamp = datetime.now().isoformat()

    error_info = {
        "timestamp": timestamp,
        "category": category.value,
        "error_type": type(error).__name__,
        "message": str(error),
        "context": context,
        "recoverable": True,
        "suggested_action": "",
    }

    # Set recovery info based on category
    if category == ErrorCategory.AUTH:
        error_info["recoverable"] = False
        error_info["suggested_action"] = "Check API key validity and permissions"
        logger.error(f"Authentication error: {error}")

    elif category == ErrorCategory.RATE_LIMIT:
        error_info["suggested_action"] = "Wait and retry with exponential backoff"
        logger.warning(f"Rate limit hit: {error}")

    elif category == ErrorCategory.VALIDATION:
        error_info["recoverable"] = False
        error_info["suggested_action"] = "Check request parameters and format"
        logger.error(f"Validation error: {error}")

    elif category == ErrorCategory.SERVER:
        error_info["suggested_action"] = "Retry after brief delay"
        logger.error(f"Server error: {error}")

    elif category == ErrorCategory.NETWORK:
        error_info["suggested_action"] = "Check network connection and retry"
        logger.warning(f"Network error: {error}")

    else:
        error_info["suggested_action"] = "Review error details and retry if appropriate"
        logger.error(f"Unknown error: {error}")

    return error_info


def safe_api_call(
    func,
    *args,
    default=None,
    on_error=None,
    **kwargs,
):
    """
    Safely execute an API call with error handling.

    Args:
        func: The function to call.
        *args: Positional arguments.
        default: Default value on error.
        on_error: Callback function on error.
        **kwargs: Keyword arguments.

    Returns:
        Function result or default value.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_info = handle_api_error(e, context=func.__name__)

        if on_error:
            on_error(error_info)

        return default


def robust_chat(prompt: str, max_retries: int = 3) -> dict:
    """
    Make a chat request with comprehensive error handling.

    Args:
        prompt: The user prompt.
        max_retries: Maximum retry attempts.

    Returns:
        Dictionary with response or error information.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                timeout=30.0,  # Set explicit timeout
            )

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "attempts": attempt + 1,
            }

        except RateLimitError as e:
            last_error = e
            wait_time = 2**attempt  # Exponential backoff
            logger.info(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
            import time

            time.sleep(wait_time)

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            if attempt < max_retries - 1:
                import time

                time.sleep(1)
            logger.warning(f"Network error, retrying (attempt {attempt + 1})")

        except InternalServerError as e:
            last_error = e
            if attempt < max_retries - 1:
                import time

                time.sleep(2)
            logger.warning(f"Server error, retrying (attempt {attempt + 1})")

        except (AuthenticationError, BadRequestError) as e:
            # Non-recoverable errors - don't retry
            return {
                "success": False,
                "error": handle_api_error(e),
                "attempts": attempt + 1,
            }

        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error: {e}")
            break

    return {
        "success": False,
        "error": handle_api_error(last_error) if last_error else None,
        "attempts": max_retries,
    }


def main():
    console.print(
        Panel.fit(
            "[bold blue]Comprehensive Error Handling[/bold blue]\nHandle API errors gracefully",
            border_style="blue",
        )
    )

    # Error types overview
    console.print("\n[bold yellow]Common API Errors:[/bold yellow]")

    errors_table = Table(show_header=True, header_style="bold cyan")
    errors_table.add_column("Error Type", style="green")
    errors_table.add_column("HTTP Code")
    errors_table.add_column("Cause")
    errors_table.add_column("Recovery")

    errors_table.add_row("AuthenticationError", "401", "Invalid API key", "Check/rotate key")
    errors_table.add_row("PermissionDeniedError", "403", "Insufficient permissions", "Check access")
    errors_table.add_row("NotFoundError", "404", "Resource not found", "Check resource ID")
    errors_table.add_row("RateLimitError", "429", "Too many requests", "Backoff and retry")
    errors_table.add_row("BadRequestError", "400", "Invalid request", "Fix request format")
    errors_table.add_row("InternalServerError", "500+", "Server issue", "Retry with delay")
    errors_table.add_row("APITimeoutError", "-", "Request timeout", "Retry or increase timeout")
    errors_table.add_row("APIConnectionError", "-", "Network issue", "Check connection")

    console.print(errors_table)

    # Example 1: Basic error handling
    console.print("\n[bold yellow]Example 1: Basic Error Handling[/bold yellow]")

    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10,
        )
        console.print(f"[green]Success:[/green] {response.choices[0].message.content}")
    except AuthenticationError:
        console.print("[red]Authentication failed. Check your API key.[/red]")
    except RateLimitError:
        console.print("[yellow]Rate limited. Please wait and retry.[/yellow]")
    except APIError as e:
        console.print(f"[red]API Error:[/red] {e}")

    # Example 2: Robust request with retries
    console.print("\n[bold yellow]Example 2: Robust Request with Retries[/bold yellow]")

    result = robust_chat("Say 'hello' in one word.")

    if result["success"]:
        console.print(f"[green]Success:[/green] {result['content']}")
        console.print(f"[dim]Completed in {result['attempts']} attempt(s)[/dim]")
    else:
        console.print(f"[red]Failed after {result['attempts']} attempts[/red]")
        if result.get("error"):
            console.print(f"[dim]Error: {result['error']['message']}[/dim]")

    # Example 3: Safe API call wrapper
    console.print("\n[bold yellow]Example 3: Safe API Call Wrapper[/bold yellow]")

    def on_error_callback(error_info):
        console.print(f"[dim]Error callback: {error_info['category']}[/dim]")

    result = safe_api_call(
        client.chat.completions.create,
        model="grok-4-1-fast-reasoning",
        messages=[{"role": "user", "content": "Hi!"}],
        default=None,
        on_error=on_error_callback,
    )

    if result:
        console.print(f"[green]Success:[/green] {result.choices[0].message.content}")
    else:
        console.print("[yellow]Request failed, using default value[/yellow]")

    # Error handling patterns
    console.print("\n[bold yellow]Error Handling Patterns:[/bold yellow]")
    console.print(
        """
  [cyan]1. Catch Specific Exceptions:[/cyan]
     Handle each error type appropriately

  [cyan]2. Exponential Backoff:[/cyan]
     Increase wait time with each retry

  [cyan]3. Circuit Breaker:[/cyan]
     Stop retrying after repeated failures

  [cyan]4. Graceful Degradation:[/cyan]
     Provide fallback behavior

  [cyan]5. Structured Logging:[/cyan]
     Log errors with context for debugging

  [cyan]6. Error Categorization:[/cyan]
     Group errors by recovery strategy
"""
    )

    # Code template
    console.print("\n[bold yellow]Error Handling Template:[/bold yellow]")
    console.print(
        """
[dim]from openai import (
    OpenAI, AuthenticationError, RateLimitError,
    APIConnectionError, APITimeoutError, InternalServerError
)

def safe_chat(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[{"role": "user", "content": prompt}]
            )
        except AuthenticationError:
            raise  # Don't retry auth errors
        except RateLimitError:
            time.sleep(2 ** attempt)  # Exponential backoff
        except (APIConnectionError, APITimeoutError):
            time.sleep(1)  # Brief delay for network issues
        except InternalServerError:
            time.sleep(2)  # Wait for server recovery
    return None  # All retries failed[/dim]
"""
    )

    # Best practices
    console.print("\n[bold yellow]Best Practices:[/bold yellow]")
    console.print(
        """
  1. [cyan]Always handle specific exceptions[/cyan]
  2. [cyan]Log errors with context[/cyan]
  3. [cyan]Set explicit timeouts[/cyan]
  4. [cyan]Use exponential backoff for retries[/cyan]
  5. [cyan]Don't retry non-recoverable errors[/cyan]
  6. [cyan]Monitor error rates in production[/cyan]
"""
    )


if __name__ == "__main__":
    main()
