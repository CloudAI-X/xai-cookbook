#!/usr/bin/env python3
"""
Grok Code Fast - Specialized for Code Tasks

Model: grok-code-fast-1
Optimized specifically for code-related tasks

Grok Code Fast excels at:
- Code generation
- Code explanation
- Bug detection and fixing
- Code review
- Refactoring suggestions
- Documentation generation
- Test generation

Best for: All code-related tasks with fast response times
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

load_dotenv()

console = Console()


def create_client() -> OpenAI:
    """Create xAI client using OpenAI SDK."""
    return OpenAI(api_key=os.environ["X_AI_API_KEY"], base_url="https://api.x.ai/v1")


def demonstrate_code_generation(client: OpenAI) -> None:
    """Demonstrate code generation capabilities."""
    console.print(
        Panel.fit(
            "[bold cyan]Grok Code Fast Demo[/bold cyan]\n"
            "Model: grok-code-fast-1 | Optimized for Code Tasks",
            border_style="cyan",
        )
    )

    prompt = """Write a Python class for a simple cache with the following features:
1. LRU (Least Recently Used) eviction policy
2. Maximum size limit
3. TTL (Time To Live) support for entries
4. Thread-safe operations

Include type hints and docstrings."""

    console.print("\n[bold]Task:[/bold] Code Generation\n")
    console.print(Panel(prompt, title="Request", border_style="yellow"))

    response = client.chat.completions.create(
        model="grok-code-fast-1",
        messages=[
            {
                "role": "system",
                "content": "You are an expert Python developer. Write clean, "
                "well-documented, production-quality code.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Generated Code:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))

    if response.usage:
        console.print(
            f"\n[dim]Tokens used - Input: {response.usage.prompt_tokens}, "
            f"Output: {response.usage.completion_tokens}[/dim]"
        )


def demonstrate_bug_detection(client: OpenAI) -> None:
    """Demonstrate bug detection and fixing."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Bug Detection and Fix\n")

    buggy_code = '''
def merge_sorted_lists(list1, list2):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1

    # Bug: Missing remaining elements
    return result


def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: Division by zero if empty list
'''

    console.print(
        Panel(
            Syntax(buggy_code, "python", theme="monokai", line_numbers=True),
            title="Code with Bugs",
            border_style="red",
        )
    )

    response = client.chat.completions.create(
        model="grok-code-fast-1",
        messages=[
            {
                "role": "system",
                "content": "You are a code reviewer. Find bugs and provide fixes with explanations.",
            },
            {
                "role": "user",
                "content": f"Find all bugs in this code and provide corrected versions:\n\n```python\n{buggy_code}\n```",
            },
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Bug Analysis and Fixes:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_code_explanation(client: OpenAI) -> None:
    """Demonstrate code explanation."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Code Explanation\n")

    complex_code = """
def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_idx = partition(arr, low, high)
        quicksort(arr, low, pivot_idx - 1)
        quicksort(arr, pivot_idx + 1, high)
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
"""

    console.print(
        Panel(
            Syntax(complex_code, "python", theme="monokai", line_numbers=True),
            title="Code to Explain",
            border_style="yellow",
        )
    )

    response = client.chat.completions.create(
        model="grok-code-fast-1",
        messages=[
            {
                "role": "system",
                "content": "You are a computer science educator. Explain code clearly "
                "with examples and visualizations where helpful.",
            },
            {
                "role": "user",
                "content": f"Explain how this quicksort implementation works step by step:\n\n```python\n{complex_code}\n```",
            },
        ],
        temperature=0.4,
        max_tokens=1500,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Code Explanation:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_test_generation(client: OpenAI) -> None:
    """Demonstrate test generation."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Test Generation\n")

    function_code = r'''
def validate_email(email: str) -> bool:
    """
    Validate an email address.

    Args:
        email: The email address to validate

    Returns:
        True if valid, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
'''

    console.print(
        Panel(
            Syntax(function_code, "python", theme="monokai", line_numbers=True),
            title="Function to Test",
            border_style="yellow",
        )
    )

    response = client.chat.completions.create(
        model="grok-code-fast-1",
        messages=[
            {
                "role": "system",
                "content": "You are a test engineer. Generate comprehensive pytest tests "
                "covering edge cases, boundary conditions, and typical usage.",
            },
            {
                "role": "user",
                "content": f"Generate pytest unit tests for this function:\n\n```python\n{function_code}\n```",
            },
        ],
        temperature=0.3,
        max_tokens=1200,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Generated Tests:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


def demonstrate_refactoring(client: OpenAI) -> None:
    """Demonstrate code refactoring suggestions."""
    console.print("\n" + "=" * 60 + "\n")
    console.print("[bold]Task:[/bold] Code Refactoring\n")

    messy_code = """
def process_data(d):
    r = []
    for i in range(len(d)):
        if d[i] > 0:
            if d[i] % 2 == 0:
                r.append(d[i] * 2)
            else:
                r.append(d[i] * 3)
    s = 0
    for x in r:
        s = s + x
    return s
"""

    console.print(
        Panel(
            Syntax(messy_code, "python", theme="monokai", line_numbers=True),
            title="Code to Refactor",
            border_style="yellow",
        )
    )

    response = client.chat.completions.create(
        model="grok-code-fast-1",
        messages=[
            {
                "role": "system",
                "content": "You are a senior developer focused on code quality. "
                "Refactor code to be more Pythonic, readable, and maintainable.",
            },
            {
                "role": "user",
                "content": f"Refactor this code to be more Pythonic and readable. "
                f"Explain your improvements:\n\n```python\n{messy_code}\n```",
            },
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    content = response.choices[0].message.content
    console.print("\n[bold green]Refactored Code:[/bold green]\n")
    console.print(Panel(Markdown(content), border_style="green"))


if __name__ == "__main__":
    client = create_client()

    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  xAI Grok Code Fast Demonstration[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")

    demonstrate_code_generation(client)
    demonstrate_bug_detection(client)
    demonstrate_code_explanation(client)
    demonstrate_test_generation(client)
    demonstrate_refactoring(client)

    console.print("\n[bold cyan]Demo complete![/bold cyan]")
    console.print("[dim]Grok Code Fast: Your specialized assistant for all coding tasks.[/dim]\n")
