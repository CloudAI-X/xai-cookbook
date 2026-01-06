#!/usr/bin/env python3
"""
03_multi_turn.py - Multi-Turn Conversation with History

This example demonstrates how to maintain conversation context across
multiple exchanges. The key is to keep track of the message history
and include it in each API call.

Key concepts:
- Maintaining conversation history
- Building multi-turn conversations
- Context persistence across turns
- Memory management for long conversations
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


class Conversation:
    """
    A class to manage multi-turn conversations with the xAI API.

    Maintains message history and handles API calls while preserving context.
    """

    def __init__(self, system_prompt: str | None = None):
        """
        Initialize a new conversation.

        Args:
            system_prompt: Optional system prompt to set the assistant's behavior.
        """
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send(self, user_message: str) -> str:
        """
        Send a message and get a response, maintaining conversation history.

        Args:
            user_message: The user's message to send.

        Returns:
            The assistant's response text.
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        # Make API call with full history
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=self.messages,
        )

        # Extract assistant message
        assistant_message = response.choices[0].message.content

        # Add assistant message to history
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def get_history(self) -> list[dict]:
        """Return the full conversation history."""
        return self.messages.copy()

    def clear_history(self, keep_system: bool = True):
        """
        Clear conversation history.

        Args:
            keep_system: If True, preserve the system prompt.
        """
        if keep_system and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []


def main():
    console.print(
        Panel.fit(
            "[bold blue]Multi-Turn Conversation Example[/bold blue]\n"
            "Maintaining context across multiple exchanges",
            border_style="blue",
        )
    )

    # Create a conversation with a system prompt
    conversation = Conversation(
        system_prompt=(
            "You are a friendly tutor helping a student learn about space. "
            "Be encouraging and build on previous topics discussed. "
            "Keep responses concise (2-3 sentences)."
        )
    )

    # Simulate a multi-turn conversation
    turns = [
        "Hi! I want to learn about the solar system.",
        "That's cool! How many planets are there?",
        "Which one is the biggest?",
        "What makes it so big?",
        "Can you remind me what we talked about first?",
    ]

    console.print("\n[bold yellow]Starting Conversation...[/bold yellow]\n")

    for i, user_input in enumerate(turns, 1):
        console.print(f"[bold green]Turn {i} - User:[/bold green] {user_input}")

        response = conversation.send(user_input)
        console.print(f"[bold cyan]Assistant:[/bold cyan] {response}\n")

    # Show the conversation history
    console.print(
        Panel.fit("[bold yellow]Conversation History[/bold yellow]", border_style="yellow")
    )

    for msg in conversation.get_history():
        role = msg["role"].upper()
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]

        if role == "SYSTEM":
            console.print(f"[dim]{role}: {content}[/dim]")
        elif role == "USER":
            console.print(f"[green]{role}: {content}[/green]")
        else:
            console.print(f"[cyan]{role}: {content}[/cyan]")

    # Demonstrate context awareness
    console.print(Panel.fit("[bold blue]Context Awareness Demo[/bold blue]", border_style="blue"))

    # New conversation showing how AI remembers context
    context_conv = Conversation()

    console.print("\n[bold green]User:[/bold green] My name is Alice and I love pizza.")
    response1 = context_conv.send("My name is Alice and I love pizza.")
    console.print(f"[bold cyan]Assistant:[/bold cyan] {response1}\n")

    console.print("[bold green]User:[/bold green] What's my name and what do I like?")
    response2 = context_conv.send("What's my name and what do I like?")
    console.print(f"[bold cyan]Assistant:[/bold cyan] {response2}")


if __name__ == "__main__":
    main()
