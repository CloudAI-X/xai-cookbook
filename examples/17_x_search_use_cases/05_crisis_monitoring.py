#!/usr/bin/env python3
"""
05_crisis_monitoring.py - Real-Time Crisis Detection and Alerting

This example demonstrates how to monitor for potential PR crises,
security incidents, or reputation threats in real-time using X posts.

Key concepts:
- Anomaly detection in mention patterns
- Negative sentiment spike detection
- Keyword-based alert triggers
- Escalation recommendation

Use cases:
- PR crisis early warning system
- Security incident detection
- Brand reputation protection
- Real-time risk monitoring
"""

import json
import os
from datetime import datetime, timedelta

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

# =============================================================================
# Constants
# =============================================================================
DEFAULT_MODEL = "grok-4-1-fast-reasoning"
MAX_SEARCH_RESULTS = 20
MAX_X_HANDLES = 10
MIN_FAVORITES_QUALITY = 50
MIN_FAVORITES_TRENDING = 100
DEFAULT_SENSITIVITY = 2.0
DEFAULT_BASELINE_HOURS = 168

# =============================================================================
# API Client Setup
# =============================================================================
api_key = os.getenv("X_AI_API_KEY")
if not api_key:
    console.print("[red]Error: X_AI_API_KEY environment variable is required.[/red]")
    console.print("Please set it in your .env file or export it:")
    console.print("  export X_AI_API_KEY=your-api-key-here")
    raise SystemExit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)


# Pydantic models for structured outputs
class CrisisMonitor(BaseModel):
    """Configuration for crisis monitoring."""

    brand_name: str = Field(description="Brand being monitored")
    crisis_keywords: list[str] = Field(description="Keywords that trigger alerts")
    baseline_sentiment: float = Field(description="Normal baseline sentiment")
    baseline_volume: int = Field(description="Normal mention volume per hour")
    monitoring_status: str = Field(description="Status: active, paused, or error")
    last_check: str = Field(description="Timestamp of last check")


class CrisisSignal(BaseModel):
    """A detected crisis signal."""

    signal_type: str = Field(
        description="Type: volume_spike, sentiment_drop, keyword_trigger, or viral_negative"
    )
    severity: str = Field(description="Severity: low, medium, high, or critical")
    trigger_reason: str = Field(description="What triggered this signal")
    affected_topics: list[str] = Field(description="Topics affected by this crisis")
    sample_posts: list[str] = Field(description="Sample posts driving the signal")
    recommended_urgency: str = Field(
        description="Response urgency: monitor, investigate, respond, or escalate"
    )
    estimated_reach: int = Field(description="Estimated users who saw related posts")
    first_detected: str = Field(description="When the signal was first detected")


class CrisisSeverityReport(BaseModel):
    """Deep analysis of a detected crisis."""

    crisis_topic: str = Field(description="The crisis topic")
    brand: str = Field(description="Affected brand")
    severity_score: float = Field(description="Severity from 0.0 to 10.0")
    affected_demographics: list[str] = Field(description="Groups most affected")
    media_coverage_risk: str = Field(
        description="Risk of mainstream media pickup: low, medium, high"
    )
    viral_velocity: str = Field(description="How fast it's spreading: slow, moderate, fast, viral")
    key_narratives: list[str] = Field(description="Main narratives being shared")
    influencer_involvement: list[str] = Field(description="Influencers discussing this crisis")
    competitor_exploitation: str = Field(description="Whether competitors are exploiting this")
    estimated_duration: str = Field(description="Expected duration of the crisis")


class CrisisResponsePlan(BaseModel):
    """Recommended response actions for a crisis."""

    immediate_actions: list[str] = Field(description="Actions to take immediately (within 1 hour)")
    communication_recommendations: list[str] = Field(description="What and how to communicate")
    stakeholders_to_notify: list[str] = Field(
        description="Internal/external stakeholders to inform"
    )
    monitoring_escalation: str = Field(description="How to escalate monitoring")
    sample_response_templates: list[str] = Field(
        description="Template responses that could be adapted"
    )
    channels_to_address: list[str] = Field(description="Channels/platforms to address")
    do_not_do: list[str] = Field(description="Actions to avoid that could worsen the crisis")


class CrisisEvolution(BaseModel):
    """How a crisis is evolving over time."""

    crisis_id: str = Field(description="Identifier for the crisis")
    current_stage: str = Field(
        description="Stage: emerging, escalating, peak, declining, or resolved"
    )
    sentiment_trajectory: str = Field(
        description="Sentiment direction: worsening, stable, or improving"
    )
    volume_trajectory: str = Field(
        description="Volume direction: increasing, stable, or decreasing"
    )
    new_developments: list[str] = Field(description="New developments since last check")
    narrative_changes: list[str] = Field(description="How the narrative has changed")
    resolution_signals: list[str] = Field(description="Signs that the crisis may be resolving")
    risk_assessment: str = Field(description="Updated risk assessment")


def get_date_range(hours: int) -> tuple[str, str]:
    """Calculate date range for analysis."""
    now = datetime.now()
    from_date = now - timedelta(hours=hours)
    return from_date.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def setup_crisis_monitor(
    brand_name: str,
    crisis_keywords: list[str],
    baseline_hours: int = 168,
) -> CrisisMonitor:
    """
    Set up crisis monitoring with custom triggers.

    Args:
        brand_name: Brand to monitor
        crisis_keywords: High-priority keywords ("breach", "recall", etc.)
        baseline_hours: Hours of history for baseline (default 7 days)

    Returns:
        CrisisMonitor configuration
    """
    # Input validation
    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    brand_name = brand_name.strip()

    if not crisis_keywords:
        raise ValueError("Crisis keywords list cannot be empty")

    # Filter out empty strings
    crisis_keywords = [k.strip() for k in crisis_keywords if k and k.strip()]
    if not crisis_keywords:
        raise ValueError("Crisis keywords list contains only empty strings")

    if baseline_hours <= 0:
        raise ValueError("baseline_hours must be a positive integer")

    keywords_str = ", ".join(crisis_keywords)

    prompt = f"""Set up crisis monitoring for {brand_name} on X.

Crisis keywords to watch: {keywords_str}

Analyze the current state and establish baselines:
1. What is the current baseline sentiment for {brand_name}?
2. What is the typical mention volume per hour?
3. Are there any current issues that need monitoring?
4. Is the monitoring system ready to detect crises?

This baseline will be used to detect anomalies."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a crisis monitoring specialist setting up early warning systems for brand protection.",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                    "return_citations": True,
                    "max_search_results": 15,
                    "sources": [
                        {
                            "type": "x",
                            "post_favorite_count": MIN_FAVORITES_QUALITY,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "crisis_monitor",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brand_name": {"type": "string"},
                            "crisis_keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "baseline_sentiment": {"type": "number"},
                            "baseline_volume": {"type": "integer"},
                            "monitoring_status": {"type": "string"},
                            "last_check": {"type": "string"},
                        },
                        "required": [
                            "brand_name",
                            "crisis_keywords",
                            "baseline_sentiment",
                            "baseline_volume",
                            "monitoring_status",
                            "last_check",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )
    except openai.APIConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise
    except openai.RateLimitError:
        console.print("[yellow]Rate limit exceeded. Please wait and try again.[/yellow]")
        raise
    except openai.APIStatusError as e:
        console.print(f"[red]API error: {e.status_code} - {e.message}[/red]")
        raise

    try:
        return CrisisMonitor.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def detect_crisis_signals(brand_name: str, sensitivity: float = 2.0) -> list[CrisisSignal]:
    """
    Detect potential crisis signals in real-time.

    Args:
        brand_name: Brand to monitor
        sensitivity: Standard deviations from baseline to trigger alert

    Returns:
        List of CrisisSignal objects
    """
    # Input validation
    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    brand_name = brand_name.strip()

    if sensitivity <= 0:
        raise ValueError("Sensitivity must be a positive number")

    prompt = f"""Scan X for potential crisis signals related to {brand_name}.

Look for:
1. Volume spikes - unusual increase in mentions (>{sensitivity}x normal)
2. Sentiment drops - significant negative sentiment shift
3. Keyword triggers - mentions of crisis-related words (breach, lawsuit, recall, scandal, etc.)
4. Viral negative content - negative posts going viral

For each signal detected, provide:
- Signal type
- Severity (low, medium, high, critical)
- What triggered it
- Affected topics
- Sample posts
- Recommended urgency (monitor, investigate, respond, escalate)
- Estimated reach
- When first detected

Be thorough but avoid false positives from normal negative feedback."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a crisis detection system identifying potential reputation threats in real-time.",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                    "return_citations": True,
                    "max_search_results": MAX_SEARCH_RESULTS,
                    "sources": [
                        {
                            "type": "x",
                            "post_favorite_count": MIN_FAVORITES_TRENDING,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "crisis_signals",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "signals": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "signal_type": {"type": "string"},
                                        "severity": {"type": "string"},
                                        "trigger_reason": {"type": "string"},
                                        "affected_topics": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "sample_posts": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "recommended_urgency": {"type": "string"},
                                        "estimated_reach": {"type": "integer"},
                                        "first_detected": {"type": "string"},
                                    },
                                    "required": [
                                        "signal_type",
                                        "severity",
                                        "trigger_reason",
                                        "affected_topics",
                                        "sample_posts",
                                        "recommended_urgency",
                                        "estimated_reach",
                                        "first_detected",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["signals"],
                        "additionalProperties": False,
                    },
                },
            },
        )
    except openai.APIConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise
    except openai.RateLimitError:
        console.print("[yellow]Rate limit exceeded. Please wait and try again.[/yellow]")
        raise
    except openai.APIStatusError as e:
        console.print(f"[red]API error: {e.status_code} - {e.message}[/red]")
        raise

    try:
        data = json.loads(response.choices[0].message.content)
        return [CrisisSignal.model_validate(s) for s in data["signals"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def analyze_crisis_severity(crisis_topic: str, brand_name: str) -> CrisisSeverityReport:
    """
    Deep analysis of a detected crisis.

    Args:
        crisis_topic: The crisis topic to analyze
        brand_name: The affected brand

    Returns:
        CrisisSeverityReport with detailed analysis
    """
    # Input validation
    if not crisis_topic or not crisis_topic.strip():
        raise ValueError("Crisis topic cannot be empty")
    crisis_topic = crisis_topic.strip()

    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    brand_name = brand_name.strip()

    prompt = f"""Provide a deep analysis of the crisis "{crisis_topic}" affecting {brand_name} on X.

Analyze:
1. Severity score (0-10 scale)
2. Which demographics/groups are most affected
3. Risk of mainstream media coverage (low/medium/high)
4. How fast it's spreading (slow/moderate/fast/viral)
5. Key narratives being shared
6. Influencers discussing this (if any)
7. Whether competitors are exploiting this
8. Expected duration of the crisis

This helps prioritize response efforts."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a crisis analyst assessing the severity and impact of brand crises.",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                    "return_citations": True,
                    "max_search_results": MAX_SEARCH_RESULTS,
                    "sources": [
                        {
                            "type": "x",
                            "post_favorite_count": MIN_FAVORITES_TRENDING,
                        },
                        {"type": "news"},
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "crisis_severity_report",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "crisis_topic": {"type": "string"},
                            "brand": {"type": "string"},
                            "severity_score": {"type": "number"},
                            "affected_demographics": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "media_coverage_risk": {"type": "string"},
                            "viral_velocity": {"type": "string"},
                            "key_narratives": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "influencer_involvement": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "competitor_exploitation": {"type": "string"},
                            "estimated_duration": {"type": "string"},
                        },
                        "required": [
                            "crisis_topic",
                            "brand",
                            "severity_score",
                            "affected_demographics",
                            "media_coverage_risk",
                            "viral_velocity",
                            "key_narratives",
                            "influencer_involvement",
                            "competitor_exploitation",
                            "estimated_duration",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )
    except openai.APIConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise
    except openai.RateLimitError:
        console.print("[yellow]Rate limit exceeded. Please wait and try again.[/yellow]")
        raise
    except openai.APIStatusError as e:
        console.print(f"[red]API error: {e.status_code} - {e.message}[/red]")
        raise

    try:
        return CrisisSeverityReport.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def generate_response_recommendations(
    crisis_topic: str, brand_name: str, severity: str
) -> CrisisResponsePlan:
    """
    Generate recommended response actions.

    Args:
        crisis_topic: The crisis topic
        brand_name: The affected brand
        severity: Severity level of the crisis

    Returns:
        CrisisResponsePlan with recommended actions
    """
    # Input validation
    if not crisis_topic or not crisis_topic.strip():
        raise ValueError("Crisis topic cannot be empty")
    crisis_topic = crisis_topic.strip()

    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    brand_name = brand_name.strip()

    if not severity or not severity.strip():
        raise ValueError("Severity level cannot be empty")
    severity = severity.strip()

    prompt = f"""Generate a crisis response plan for {brand_name} regarding "{crisis_topic}".

Severity level: {severity}

Provide:
1. Immediate actions (within 1 hour)
2. Communication recommendations
3. Stakeholders to notify (internal and external)
4. How to escalate monitoring
5. Sample response templates (2-3 that can be adapted)
6. Channels/platforms to address
7. Things to avoid that could worsen the crisis

Focus on practical, actionable recommendations based on crisis communications best practices."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a crisis communications expert providing response recommendations.",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                    "return_citations": True,
                    "max_search_results": 10,
                    "sources": [
                        {
                            "type": "x",
                            "post_favorite_count": MIN_FAVORITES_QUALITY,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "crisis_response_plan",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "immediate_actions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "communication_recommendations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "stakeholders_to_notify": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "monitoring_escalation": {"type": "string"},
                            "sample_response_templates": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "channels_to_address": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "do_not_do": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "immediate_actions",
                            "communication_recommendations",
                            "stakeholders_to_notify",
                            "monitoring_escalation",
                            "sample_response_templates",
                            "channels_to_address",
                            "do_not_do",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )
    except openai.APIConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise
    except openai.RateLimitError:
        console.print("[yellow]Rate limit exceeded. Please wait and try again.[/yellow]")
        raise
    except openai.APIStatusError as e:
        console.print(f"[red]API error: {e.status_code} - {e.message}[/red]")
        raise

    try:
        return CrisisResponsePlan.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def track_crisis_evolution(crisis_topic: str, brand_name: str, hours: int = 24) -> CrisisEvolution:
    """
    Track how a crisis is evolving over time.

    Args:
        crisis_topic: The crisis to track
        brand_name: The affected brand
        hours: Hours of evolution to track

    Returns:
        CrisisEvolution with current state and trajectory
    """
    # Input validation
    if not crisis_topic or not crisis_topic.strip():
        raise ValueError("Crisis topic cannot be empty")
    crisis_topic = crisis_topic.strip()

    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    brand_name = brand_name.strip()

    if hours <= 0:
        raise ValueError("Hours must be a positive integer")

    from_date, to_date = get_date_range(hours)

    prompt = f"""Track the evolution of the crisis "{crisis_topic}" for {brand_name} over the past {hours} hours.

Analyze:
1. Current stage (emerging, escalating, peak, declining, resolved)
2. Sentiment trajectory (worsening, stable, improving)
3. Volume trajectory (increasing, stable, decreasing)
4. New developments since it started
5. How the narrative has changed
6. Any signs of resolution
7. Updated risk assessment

This helps determine if response efforts are working."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a crisis evolution tracker monitoring how brand crises develop over time.",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                    "return_citations": True,
                    "max_search_results": MAX_SEARCH_RESULTS,
                    "from_date": from_date,
                    "to_date": to_date,
                    "sources": [
                        {
                            "type": "x",
                            "post_favorite_count": MIN_FAVORITES_QUALITY,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "crisis_evolution",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "crisis_id": {"type": "string"},
                            "current_stage": {"type": "string"},
                            "sentiment_trajectory": {"type": "string"},
                            "volume_trajectory": {"type": "string"},
                            "new_developments": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "narrative_changes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "resolution_signals": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "risk_assessment": {"type": "string"},
                        },
                        "required": [
                            "crisis_id",
                            "current_stage",
                            "sentiment_trajectory",
                            "volume_trajectory",
                            "new_developments",
                            "narrative_changes",
                            "resolution_signals",
                            "risk_assessment",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )
    except openai.APIConnectionError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise
    except openai.RateLimitError:
        console.print("[yellow]Rate limit exceeded. Please wait and try again.[/yellow]")
        raise
    except openai.APIStatusError as e:
        console.print(f"[red]API error: {e.status_code} - {e.message}[/red]")
        raise

    try:
        return CrisisEvolution.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    return {
        "critical": "red bold",
        "high": "red",
        "medium": "yellow",
        "low": "green",
    }.get(severity.lower(), "white")


def get_stage_color(stage: str) -> str:
    """Get color for crisis stage."""
    return {
        "emerging": "yellow",
        "escalating": "red",
        "peak": "red bold",
        "declining": "green",
        "resolved": "green bold",
    }.get(stage.lower(), "white")


def main():
    console.print(
        Panel.fit(
            "[bold blue]Real-Time Crisis Detection and Alerting[/bold blue]\n"
            "Monitor for potential PR crises and reputation threats on X",
            border_style="blue",
        )
    )

    # Example 1: Set up crisis monitoring
    console.print("\n[bold yellow]Example 1: Crisis Monitoring Setup[/bold yellow]")
    console.print("[dim]Setting up crisis monitoring for TechCorp[/dim]")

    monitor = setup_crisis_monitor(
        brand_name="Apple",
        crisis_keywords=[
            "data breach",
            "lawsuit",
            "recall",
            "privacy",
            "security flaw",
        ],
        baseline_hours=168,
    )

    content = f"""
[bold]Brand:[/bold] {monitor.brand_name}
[bold]Status:[/bold] [green]{monitor.monitoring_status.upper()}[/green]
[bold]Baseline Sentiment:[/bold] {monitor.baseline_sentiment:.2f}
[bold]Baseline Volume:[/bold] {monitor.baseline_volume}/hour
[bold]Last Check:[/bold] {monitor.last_check}

[bold cyan]Crisis Keywords Monitored:[/bold cyan]
"""
    for keyword in monitor.crisis_keywords:
        content += f"  - {keyword}\n"

    console.print(Panel(content, title="Crisis Monitor Configuration", border_style="green"))

    # Example 2: Detect crisis signals
    console.print("\n[bold yellow]Example 2: Crisis Signal Detection[/bold yellow]")
    console.print("[dim]Scanning for potential crisis signals[/dim]")

    signals = detect_crisis_signals("Microsoft", sensitivity=2.0)

    if signals:
        for signal in signals[:3]:
            severity_color = get_severity_color(signal.severity)
            color_base = severity_color.split()[0]

            urgency_colors = {
                "escalate": "red bold",
                "respond": "red",
                "investigate": "yellow",
                "monitor": "green",
            }
            urgency_color = urgency_colors.get(signal.recommended_urgency.lower(), "white")

            content = f"""
[bold]Signal Type:[/bold] {signal.signal_type}
[bold]Severity:[/bold] [{severity_color}]{signal.severity.upper()}[/{severity_color}]
[bold]Trigger:[/bold] {signal.trigger_reason}
[bold]Estimated Reach:[/bold] {signal.estimated_reach:,}
[bold]First Detected:[/bold] {signal.first_detected}
[bold]Urgency:[/bold] [{urgency_color}]{signal.recommended_urgency.upper()}[/{urgency_color}]

[bold cyan]Affected Topics:[/bold cyan]
"""
            for topic in signal.affected_topics[:3]:
                content += f"  - {topic}\n"

            content += "\n[bold cyan]Sample Posts:[/bold cyan]\n"
            for post in signal.sample_posts[:2]:
                content += f"  - {post[:80]}...\n"

            console.print(
                Panel(
                    content,
                    title=f"CRISIS SIGNAL: {signal.signal_type.upper()}",
                    border_style=color_base,
                )
            )
    else:
        console.print("[green]No crisis signals detected. Brand reputation appears stable.[/green]")

    # Example 3: Analyze crisis severity
    console.print("\n[bold yellow]Example 3: Crisis Severity Analysis[/bold yellow]")
    console.print("[dim]Deep analysis of a detected crisis[/dim]")

    severity_report = analyze_crisis_severity(
        crisis_topic="Data privacy concerns",
        brand_name="Meta",
    )

    severity_color = (
        "red"
        if severity_report.severity_score > 7
        else ("yellow" if severity_report.severity_score > 4 else "green")
    )

    content = f"""
[bold]Crisis:[/bold] {severity_report.crisis_topic}
[bold]Brand:[/bold] {severity_report.brand}
[bold]Severity Score:[/bold] [{severity_color}]{severity_report.severity_score:.1f}/10[/{severity_color}]
[bold]Media Coverage Risk:[/bold] {severity_report.media_coverage_risk}
[bold]Viral Velocity:[/bold] {severity_report.viral_velocity}
[bold]Competitor Exploitation:[/bold] {severity_report.competitor_exploitation}
[bold]Estimated Duration:[/bold] {severity_report.estimated_duration}

[bold cyan]Affected Demographics:[/bold cyan]
"""
    for demo in severity_report.affected_demographics[:3]:
        content += f"  - {demo}\n"

    content += "\n[bold cyan]Key Narratives:[/bold cyan]\n"
    for narrative in severity_report.key_narratives[:3]:
        content += f"  - {narrative}\n"

    if severity_report.influencer_involvement:
        content += "\n[bold cyan]Influencer Involvement:[/bold cyan]\n"
        for inf in severity_report.influencer_involvement[:3]:
            content += f"  - {inf}\n"

    console.print(
        Panel(
            content,
            title="Crisis Severity Report",
            border_style=severity_color,
        )
    )

    # Example 4: Generate response plan
    console.print("\n[bold yellow]Example 4: Crisis Response Plan[/bold yellow]")
    console.print("[dim]Generating response recommendations[/dim]")

    response_plan = generate_response_recommendations(
        crisis_topic="Service outage complaints",
        brand_name="AWS",
        severity="high",
    )

    content = """
[bold red]IMMEDIATE ACTIONS (Within 1 Hour):[/bold red]
"""
    for i, action in enumerate(response_plan.immediate_actions[:4], 1):
        content += f"  {i}. {action}\n"

    content += "\n[bold cyan]COMMUNICATION RECOMMENDATIONS:[/bold cyan]\n"
    for rec in response_plan.communication_recommendations[:3]:
        content += f"  - {rec}\n"

    content += "\n[bold cyan]STAKEHOLDERS TO NOTIFY:[/bold cyan]\n"
    for stakeholder in response_plan.stakeholders_to_notify[:4]:
        content += f"  - {stakeholder}\n"

    content += f"\n[bold cyan]MONITORING ESCALATION:[/bold cyan]\n  {response_plan.monitoring_escalation}\n"

    content += "\n[bold red]DO NOT:[/bold red]\n"
    for dont in response_plan.do_not_do[:3]:
        content += f"  - {dont}\n"

    console.print(Panel(content, title="Crisis Response Plan", border_style="red"))

    # Show sample response templates
    console.print("\n[bold cyan]Sample Response Templates:[/bold cyan]")
    for i, template in enumerate(response_plan.sample_response_templates[:2], 1):
        console.print(Panel(template, title=f"Template {i}", border_style="cyan"))

    # Example 5: Track crisis evolution
    console.print("\n[bold yellow]Example 5: Crisis Evolution Tracking[/bold yellow]")
    console.print("[dim]Tracking how a crisis is evolving[/dim]")

    evolution = track_crisis_evolution(
        crisis_topic="Product quality issues",
        brand_name="Tesla",
        hours=24,
    )

    stage_color = get_stage_color(evolution.current_stage)

    # Trajectory indicators
    sentiment_indicator = {
        "worsening": "[red]WORSENING[/red]",
        "stable": "[yellow]STABLE[/yellow]",
        "improving": "[green]IMPROVING[/green]",
    }.get(evolution.sentiment_trajectory.lower(), evolution.sentiment_trajectory)

    volume_indicator = {
        "increasing": "[red]INCREASING[/red]",
        "stable": "[yellow]STABLE[/yellow]",
        "decreasing": "[green]DECREASING[/green]",
    }.get(evolution.volume_trajectory.lower(), evolution.volume_trajectory)

    content = f"""
[bold]Crisis ID:[/bold] {evolution.crisis_id}
[bold]Current Stage:[/bold] [{stage_color}]{evolution.current_stage.upper()}[/{stage_color}]
[bold]Sentiment Trajectory:[/bold] {sentiment_indicator}
[bold]Volume Trajectory:[/bold] {volume_indicator}
[bold]Risk Assessment:[/bold] {evolution.risk_assessment}

[bold cyan]New Developments:[/bold cyan]
"""
    for dev in evolution.new_developments[:3]:
        content += f"  - {dev}\n"

    content += "\n[bold cyan]Narrative Changes:[/bold cyan]\n"
    for change in evolution.narrative_changes[:3]:
        content += f"  - {change}\n"

    if evolution.resolution_signals:
        content += "\n[bold green]Resolution Signals:[/bold green]\n"
        for signal in evolution.resolution_signals[:3]:
            content += f"  - {signal}\n"

    console.print(Panel(content, title="Crisis Evolution", border_style=stage_color.split()[0]))

    # Severity level reference
    console.print("\n[bold yellow]Crisis Severity Reference:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Level", style="bold")
    ref_table.add_column("Score Range")
    ref_table.add_column("Response Time")
    ref_table.add_column("Escalation")

    ref_table.add_row(
        "[green]Low[/green]",
        "0-3",
        "24-48 hours",
        "Monitor only",
    )
    ref_table.add_row(
        "[yellow]Medium[/yellow]",
        "4-6",
        "4-8 hours",
        "Manager notification",
    )
    ref_table.add_row(
        "[red]High[/red]",
        "7-8",
        "1-2 hours",
        "Executive notification",
    )
    ref_table.add_row(
        "[red bold]Critical[/red bold]",
        "9-10",
        "Immediate",
        "War room activation",
    )

    console.print(ref_table)

    console.print("\n[bold yellow]Crisis Detection Parameters:[/bold yellow]")

    param_table = Table(show_header=True, header_style="bold cyan")
    param_table.add_column("Parameter", style="green")
    param_table.add_column("Description")
    param_table.add_column("Recommended Value")

    param_table.add_row(
        "post_favorite_count",
        "Filter for impactful posts",
        "100+ for crisis detection",
    )
    param_table.add_row(
        "max_search_results",
        "Posts to analyze",
        "20 for comprehensive scanning",
    )
    param_table.add_row(
        "sources",
        "Include news for media pickup",
        '["x", "news"]',
    )

    console.print(param_table)


if __name__ == "__main__":
    main()
