#!/usr/bin/env python3
"""
01_voice_overview.py - Voice Agent Overview

This example provides an overview of xAI's Voice Agent capabilities.
Voice agents enable real-time spoken conversations with Grok.

Note: Voice API functionality may require specific access and setup.
Check docs.x.ai for the latest availability and implementation details.

Key concepts:
- Voice agent architecture
- Real-time audio streaming
- Speech-to-text and text-to-speech
- Voice conversation patterns
"""

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

# xAI API configuration
API_KEY = os.environ.get("X_AI_API_KEY")


def main():
    console.print(
        Panel.fit(
            "[bold blue]Voice Agent Overview[/bold blue]\nReal-time spoken conversations with Grok",
            border_style="blue",
        )
    )

    # Voice agent overview
    console.print("\n[bold yellow]What are Voice Agents?[/bold yellow]")
    console.print(
        """
  Voice agents enable [cyan]spoken conversations[/cyan] with Grok, allowing
  users to interact naturally using their voice. The system handles:

  - [green]Speech Recognition:[/green] Converting audio to text
  - [green]Natural Language:[/green] Processing user intent
  - [green]Response Generation:[/green] Creating appropriate replies
  - [green]Speech Synthesis:[/green] Converting text back to audio
"""
    )

    # Architecture diagram
    console.print("\n[bold yellow]Voice Agent Architecture:[/bold yellow]")
    console.print(
        """
  [dim]
  +----------------+     +----------------+     +----------------+
  |   User Audio   | --> | Speech-to-Text | --> |   Grok Model   |
  |   (Microphone) |     |   (STT/ASR)    |     |   Processing   |
  +----------------+     +----------------+     +----------------+
                                                        |
                                                        v
  +----------------+     +----------------+     +----------------+
  | Audio Playback | <-- | Text-to-Speech | <-- |    Response    |
  |   (Speaker)    |     |     (TTS)      |     |   Generation   |
  +----------------+     +----------------+     +----------------+
  [/dim]
"""
    )

    # Key features
    console.print("\n[bold yellow]Key Features:[/bold yellow]")

    features_table = Table(show_header=True, header_style="bold cyan")
    features_table.add_column("Feature", style="green")
    features_table.add_column("Description")
    features_table.add_column("Use Case")

    features_table.add_row(
        "Real-time Streaming",
        "Low-latency audio processing",
        "Interactive conversations",
    )
    features_table.add_row(
        "Multi-turn Dialogue",
        "Maintains conversation context",
        "Complex discussions",
    )
    features_table.add_row(
        "Tool Integration",
        "Voice-triggered actions",
        "Voice commands, queries",
    )
    features_table.add_row(
        "Natural Voices",
        "Human-like speech synthesis",
        "Engaging user experience",
    )

    console.print(features_table)

    # Implementation patterns
    console.print("\n[bold yellow]Implementation Patterns:[/bold yellow]")
    console.print(
        """
  [cyan]1. WebSocket Connection[/cyan]
     Voice agents typically use WebSocket for real-time bidirectional
     audio streaming between client and server.

  [cyan]2. Audio Chunking[/cyan]
     Audio is processed in chunks for low-latency responses.
     Common formats: PCM 16-bit, 16kHz sample rate.

  [cyan]3. Turn Detection[/cyan]
     System detects when user stops speaking to begin processing.
     Can be voice activity detection (VAD) or push-to-talk.

  [cyan]4. Interruption Handling[/cyan]
     Allow users to interrupt the agent mid-response for
     natural conversation flow.
"""
    )

    # Example architecture (conceptual)
    console.print("\n[bold yellow]Conceptual Code Pattern:[/bold yellow]")
    console.print(
        """
[dim]# Note: This is a conceptual example
# Check docs.x.ai for actual implementation

import asyncio
import websockets

async def voice_agent():
    # Connect to voice API
    async with websockets.connect(
        "wss://api.x.ai/v1/voice",
        extra_headers={"Authorization": f"Bearer {api_key}"}
    ) as ws:

        # Send configuration
        await ws.send(json.dumps({
            "type": "config",
            "model": "grok-voice",
            "voice": "default",
            "sample_rate": 16000,
        }))

        # Audio streaming loop
        async def send_audio():
            while True:
                audio_chunk = await get_microphone_audio()
                await ws.send(audio_chunk)

        async def receive_audio():
            while True:
                response = await ws.recv()
                if isinstance(response, bytes):
                    play_audio(response)
                else:
                    data = json.loads(response)
                    if data["type"] == "transcript":
                        print(f"User: {data['text']}")
                    elif data["type"] == "response":
                        print(f"Grok: {data['text']}")

        await asyncio.gather(send_audio(), receive_audio())

# Run the voice agent
asyncio.run(voice_agent())[/dim]
"""
    )

    # Use cases
    console.print("\n[bold yellow]Common Use Cases:[/bold yellow]")

    use_cases_table = Table(show_header=True, header_style="bold cyan")
    use_cases_table.add_column("Application", style="green")
    use_cases_table.add_column("Description")

    use_cases_table.add_row(
        "Customer Support",
        "Voice-based customer service agents",
    )
    use_cases_table.add_row(
        "Virtual Assistants",
        "Hands-free productivity helpers",
    )
    use_cases_table.add_row(
        "Accessibility",
        "Voice interfaces for accessibility needs",
    )
    use_cases_table.add_row(
        "Smart Devices",
        "Voice control for IoT and smart home",
    )
    use_cases_table.add_row(
        "Language Learning",
        "Conversational practice partners",
    )
    use_cases_table.add_row(
        "Healthcare",
        "Voice-based health check-ins",
    )

    console.print(use_cases_table)

    # Requirements
    console.print("\n[bold yellow]Technical Requirements:[/bold yellow]")

    reqs_table = Table(show_header=True, header_style="bold cyan")
    reqs_table.add_column("Component", style="green")
    reqs_table.add_column("Requirement")

    reqs_table.add_row(
        "Audio Input",
        "Microphone access (browser or native)",
    )
    reqs_table.add_row(
        "Audio Output",
        "Speaker/headphone playback capability",
    )
    reqs_table.add_row(
        "Network",
        "Stable connection for WebSocket streaming",
    )
    reqs_table.add_row(
        "Format",
        "PCM audio (typically 16kHz, 16-bit)",
    )

    console.print(reqs_table)

    # Getting started
    console.print("\n[bold yellow]Getting Started:[/bold yellow]")
    console.print(
        """
  1. [cyan]Check Access:[/cyan]
     Voice agents may require specific API access.
     Visit docs.x.ai to check availability for your account.

  2. [cyan]Review Documentation:[/cyan]
     The Voice API documentation includes:
     - WebSocket endpoint details
     - Audio format specifications
     - Configuration options
     - SDK examples

  3. [cyan]Test Environment:[/cyan]
     - Ensure microphone permissions
     - Test audio playback
     - Check network latency

  4. [cyan]Implementation:[/cyan]
     - Use official xAI SDK if available
     - Or implement WebSocket client directly
     - Handle audio encoding/decoding
"""
    )

    # Resources
    console.print("\n[bold yellow]Resources:[/bold yellow]")
    console.print(
        """
  - [cyan]Documentation:[/cyan] https://docs.x.ai/docs/guides/voice
  - [cyan]API Reference:[/cyan] https://docs.x.ai/docs/api-reference
  - [cyan]xAI Console:[/cyan] https://console.x.ai

  [dim]Note: Voice API availability and features may vary.
  Always check the official documentation for current details.[/dim]
"""
    )

    # Status
    console.print("\n[bold yellow]Current Status:[/bold yellow]")
    if API_KEY:
        console.print("[green]API key configured.[/green] Check docs.x.ai for Voice API access.")
    else:
        console.print("[yellow]API key not set.[/yellow] Set X_AI_API_KEY environment variable.")


if __name__ == "__main__":
    main()
