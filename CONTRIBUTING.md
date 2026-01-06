# Contributing to xAI Cookbook

Thank you for your interest in contributing to the xAI Cookbook! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Adding New Examples](#adding-new-examples)
- [Submitting Changes](#submitting-changes)
- [Getting Help](#getting-help)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/xai-cookbook.git
   cd xai-cookbook
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/CloudAI-X/xai-cookbook.git
   ```

## Development Setup

We use [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- xAI API key from [console.x.ai](https://console.x.ai)

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up your environment
cp .env.example .env
# Edit .env and add your X_AI_API_KEY
```

### Running Examples

```bash
# Run a specific example
uv run python examples/02_chat_completions/01_basic_chat.py

# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy.readthedocs.io/) for type checking.

### Before Submitting

Always run these commands before submitting a PR:

```bash
# Fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Check types
uv run mypy src/
```

### Style Guidelines

- **Line length**: 100 characters (configured in pyproject.toml)
- **Imports**: Sorted with isort (handled by ruff)
- **Docstrings**: Google style for all public functions
- **Type hints**: Required for function signatures
- **Comments**: Use sparingly; code should be self-documenting

## Adding New Examples

### Example Template

Each example should follow this structure:

```python
#!/usr/bin/env python3
"""
XX_example_name.py - Brief Description

This example demonstrates [what it does].

Key concepts:
- Concept 1
- Concept 2
- Concept 3
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1",
)

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold blue]Example Title[/bold blue]\n"
            "Brief description",
            border_style="blue",
        )
    )

    # Example implementation
    # ...


if __name__ == "__main__":
    main()
```

### Guidelines for Examples

1. **Self-contained**: Each example should run independently
2. **Educational**: Include comments explaining key concepts
3. **Rich output**: Use the `rich` library for formatted console output
4. **Error handling**: Handle common errors gracefully
5. **Docstrings**: Document what the example demonstrates

### Example Categories

Place your example in the appropriate category:

| Category                 | Description              |
| ------------------------ | ------------------------ |
| `01_getting_started/`    | Setup and first steps    |
| `02_chat_completions/`   | Basic chat functionality |
| `03_models_showcase/`    | Model comparisons        |
| `04_vision/`             | Image understanding      |
| `05_image_generation/`   | Creating images          |
| `06_function_calling/`   | Tool use and functions   |
| `07_server_side_tools/`  | Web search, code exec    |
| `08_structured_outputs/` | JSON and schema outputs  |
| `09_live_search/`        | Real-time search         |
| `10_files/`              | File upload and chat     |
| `11_collections/`        | Knowledge base and RAG   |
| `12_voice/`              | Voice agent overview     |
| `13_utilities/`          | Helper utilities         |
| `14_management_api/`     | API keys and usage       |
| `15_anthropic_compat/`   | Anthropic API compat     |
| `16_advanced/`           | Production patterns      |

## Submitting Changes

### Pull Request Process

1. **Create a branch** for your changes:

   ```bash
   git checkout -b feature/my-new-example
   ```

2. **Make your changes** and commit:

   ```bash
   git add .
   git commit -m "Add example for [feature]"
   ```

3. **Keep your branch updated**:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push to your fork**:

   ```bash
   git push origin feature/my-new-example
   ```

5. **Open a Pull Request** on GitHub

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issues if applicable ("Fix #123")

Examples:

- `Add streaming example with progress bar`
- `Fix token counting in utilities`
- `Update README with new examples`

### Review Process

1. All PRs require at least one review
2. CI checks must pass (linting, type checking)
3. Examples must run successfully with a valid API key

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/CloudAI-X/xai-cookbook/discussions)
- **Bugs**: Open an [Issue](https://github.com/CloudAI-X/xai-cookbook/issues)
- **xAI API Docs**: [docs.x.ai](https://docs.x.ai)

---

Thank you for contributing to the xAI Cookbook!
