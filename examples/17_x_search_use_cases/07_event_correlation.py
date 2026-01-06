#!/usr/bin/env python3
"""
07_event_correlation.py - Event Impact and X Reaction Analysis

This example demonstrates how to correlate real-world events with X reactions
and measure their impact on social media discussions.

Key concepts:
- Event detection from news sources
- X reaction correlation
- Impact measurement
- Narrative tracking

Use cases:
- Event impact assessment
- PR campaign analysis
- Product launch monitoring
- Crisis response measurement
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
DEFAULT_HOURS_SINCE_EVENT = 24
DEFAULT_EVOLUTION_HOURS = 48

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
class EventReactionAnalysis(BaseModel):
    """Analysis of X's reaction to an event."""

    event: str = Field(description="Description of the event")
    total_x_mentions: int = Field(description="Total mentions on X")
    peak_hour: str = Field(description="When mentions peaked")
    sentiment_distribution: dict[str, float] = Field(
        description="Percentage breakdown of sentiments"
    )
    key_narratives: list[str] = Field(description="Main narratives being shared")
    most_influential_reactions: list[str] = Field(
        description="Most impactful reactions from influencers"
    )
    mainstream_vs_grassroots: dict[str, float] = Field(
        description="Percentage from verified vs regular accounts"
    )
    geographic_spread: list[str] = Field(description="Key regions discussing the event")


class NewsXCorrelation(BaseModel):
    """Correlation between news coverage and X discussions."""

    topic: str = Field(description="Topic being analyzed")
    news_volume: str = Field(description="Level of news coverage")
    x_volume: str = Field(description="Level of X discussion")
    correlation_strength: str = Field(description="How correlated they are: weak, moderate, strong")
    leading_source: str = Field(description="Which led: news, x, or simultaneous")
    time_lag: str = Field(description="Time between news and X reaction")
    narrative_differences: list[str] = Field(description="How narratives differ between news and X")
    key_news_outlets: list[str] = Field(description="Major outlets covering this")
    x_reaction_summary: str = Field(description="Summary of X reaction")


class NarrativeEvolution(BaseModel):
    """How narratives evolve on X after an event."""

    event_topic: str = Field(description="The event topic")
    initial_narrative: str = Field(description="The initial narrative")
    evolved_narratives: list[str] = Field(description="How the narrative evolved")
    narrative_shifts: list[dict] = Field(description="Key narrative shifts with timing")
    dominant_narrative: str = Field(description="The currently dominant narrative")
    counter_narratives: list[str] = Field(description="Opposing narratives")
    influencer_impact: list[str] = Field(description="How influencers shaped the narrative")


class EventImpactReport(BaseModel):
    """Comprehensive event impact analysis."""

    event_description: str = Field(description="What the event was")
    affected_entities: list[str] = Field(description="Entities affected by the event")
    overall_impact_score: float = Field(description="Impact score from 0 to 10")
    reach_estimate: int = Field(description="Estimated users who saw related content")
    sentiment_impact: dict[str, float] = Field(description="Sentiment change for each entity")
    brand_implications: list[str] = Field(description="Implications for affected brands")
    recommended_responses: list[str] = Field(description="Recommended responses for each entity")
    comparison_to_similar_events: str = Field(
        description="How this compares to similar past events"
    )


class EventLongevityPrediction(BaseModel):
    """Prediction for how long an event will remain in discussion."""

    event_topic: str = Field(description="The event topic")
    current_momentum: str = Field(
        description="Current momentum: building, peak, declining, or fading"
    )
    predicted_duration: str = Field(description="Expected discussion duration")
    factors_extending: list[str] = Field(description="Factors that could extend discussion")
    factors_shortening: list[str] = Field(description="Factors that could shorten discussion")
    similar_event_comparison: str = Field(description="Duration of similar past events")
    recommended_timing: str = Field(description="Best time to make related announcements")


def get_date_range(hours: int) -> tuple[str, str]:
    """Calculate date range for analysis."""
    now = datetime.now()
    from_date = now - timedelta(hours=hours)
    return from_date.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def analyze_event_reaction(
    event_description: str, hours_since_event: int = 24
) -> EventReactionAnalysis:
    """
    Analyze X's reaction to a specific event.

    Args:
        event_description: What happened
        hours_since_event: How long ago the event occurred

    Returns:
        EventReactionAnalysis with comprehensive analysis
    """
    # Input validation
    if not event_description or not event_description.strip():
        raise ValueError("Event description cannot be empty")
    event_description = event_description.strip()

    if hours_since_event <= 0:
        raise ValueError("hours_since_event must be a positive integer")

    from_date, to_date = get_date_range(hours_since_event)

    prompt = f"""Analyze X's reaction to this event: "{event_description}"

The event occurred within the last {hours_since_event} hours.

Provide:
1. Total estimated X mentions about this event
2. When mentions peaked (time of day)
3. Sentiment breakdown (% positive, negative, neutral)
4. Key narratives being shared (3-5 main ones)
5. Most influential reactions (from verified/notable accounts)
6. Breakdown of mainstream (verified) vs grassroots (regular) users
7. Key geographic regions discussing this

Focus on understanding the overall public reaction pattern."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an event analyst studying public reactions on social media.",
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
                            "post_favorite_count": MIN_FAVORITES_TRENDING,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "event_reaction_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "event": {"type": "string"},
                            "total_x_mentions": {"type": "integer"},
                            "peak_hour": {"type": "string"},
                            "sentiment_distribution": {
                                "type": "object",
                                "properties": {
                                    "positive": {"type": "number"},
                                    "negative": {"type": "number"},
                                    "neutral": {"type": "number"},
                                },
                                "required": ["positive", "negative", "neutral"],
                                "additionalProperties": False,
                            },
                            "key_narratives": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "most_influential_reactions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "mainstream_vs_grassroots": {
                                "type": "object",
                                "properties": {
                                    "mainstream": {"type": "number"},
                                    "grassroots": {"type": "number"},
                                },
                                "required": ["mainstream", "grassroots"],
                                "additionalProperties": False,
                            },
                            "geographic_spread": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "event",
                            "total_x_mentions",
                            "peak_hour",
                            "sentiment_distribution",
                            "key_narratives",
                            "most_influential_reactions",
                            "mainstream_vs_grassroots",
                            "geographic_spread",
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
        return EventReactionAnalysis.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def correlate_news_and_social(topic: str) -> NewsXCorrelation:
    """
    Compare news coverage with X discussions.

    Args:
        topic: Topic to analyze

    Returns:
        NewsXCorrelation with comparison analysis
    """
    # Input validation
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")
    topic = topic.strip()

    prompt = f"""Compare news coverage and X discussions about: "{topic}"

Analyze:
1. Volume of news coverage (low, moderate, high)
2. Volume of X discussion (low, moderate, high)
3. How strongly they correlate (weak, moderate, strong)
4. Which source led the conversation (news, X, or simultaneous)
5. Time lag between news and X reaction
6. How narratives differ between news and X
7. Key news outlets covering this
8. Summary of X reaction

This helps understand the relationship between traditional and social media."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a media analyst comparing news and social media coverage patterns.",
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
                    "name": "news_x_correlation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "news_volume": {"type": "string"},
                            "x_volume": {"type": "string"},
                            "correlation_strength": {"type": "string"},
                            "leading_source": {"type": "string"},
                            "time_lag": {"type": "string"},
                            "narrative_differences": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "key_news_outlets": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "x_reaction_summary": {"type": "string"},
                        },
                        "required": [
                            "topic",
                            "news_volume",
                            "x_volume",
                            "correlation_strength",
                            "leading_source",
                            "time_lag",
                            "narrative_differences",
                            "key_news_outlets",
                            "x_reaction_summary",
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
        return NewsXCorrelation.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def track_narrative_evolution(event_topic: str, hours: int = 48) -> NarrativeEvolution:
    """
    Track how narratives evolve on X after an event.

    Args:
        event_topic: The event to track
        hours: Hours of evolution to analyze

    Returns:
        NarrativeEvolution with narrative tracking
    """
    # Input validation
    if not event_topic or not event_topic.strip():
        raise ValueError("Event topic cannot be empty")
    event_topic = event_topic.strip()

    if hours <= 0:
        raise ValueError("hours must be a positive integer")

    from_date, to_date = get_date_range(hours)

    prompt = f"""Track how narratives about "{event_topic}" evolved on X over the past {hours} hours.

Analyze:
1. What was the initial narrative when it started?
2. How has the narrative evolved? (list key stages)
3. Key narrative shifts (with approximate timing)
4. What is the currently dominant narrative?
5. What counter-narratives exist?
6. How have influencers shaped the narrative?

This helps understand how public opinion forms and changes."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a narrative analyst tracking how public discussions evolve over time.",
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
                            "post_favorite_count": MIN_FAVORITES_TRENDING,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "narrative_evolution",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "event_topic": {"type": "string"},
                            "initial_narrative": {"type": "string"},
                            "evolved_narratives": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "narrative_shifts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "time": {"type": "string"},
                                        "from_narrative": {"type": "string"},
                                        "to_narrative": {"type": "string"},
                                        "trigger": {"type": "string"},
                                    },
                                    "required": [
                                        "time",
                                        "from_narrative",
                                        "to_narrative",
                                        "trigger",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "dominant_narrative": {"type": "string"},
                            "counter_narratives": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "influencer_impact": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "event_topic",
                            "initial_narrative",
                            "evolved_narratives",
                            "narrative_shifts",
                            "dominant_narrative",
                            "counter_narratives",
                            "influencer_impact",
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
        return NarrativeEvolution.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def measure_event_impact(
    event_keywords: list[str], affected_entities: list[str]
) -> EventImpactReport:
    """
    Measure the social impact of an event.

    Args:
        event_keywords: Keywords describing the event
        affected_entities: Brands/people affected by the event

    Returns:
        EventImpactReport with comprehensive impact analysis
    """
    # Input validation
    if not event_keywords:
        raise ValueError("event_keywords list cannot be empty")
    # Filter out empty strings and strip whitespace
    event_keywords = [k.strip() for k in event_keywords if k and k.strip()]
    if not event_keywords:
        raise ValueError("event_keywords must contain at least one non-empty keyword")

    if not affected_entities:
        raise ValueError("affected_entities list cannot be empty")
    # Filter out empty strings and strip whitespace
    affected_entities = [e.strip() for e in affected_entities if e and e.strip()]
    if not affected_entities:
        raise ValueError("affected_entities must contain at least one non-empty entity")

    keywords_str = ", ".join(event_keywords)
    entities_str = ", ".join(affected_entities)

    prompt = f"""Measure the impact of this event on X:
Event keywords: {keywords_str}
Affected entities: {entities_str}

Analyze:
1. Brief description of the event
2. Overall impact score (0-10)
3. Estimated reach (users who saw related content)
4. Sentiment change for each affected entity (-1.0 to 1.0 change)
5. Brand implications for affected entities
6. Recommended responses for each entity
7. How this compares to similar past events

Provide actionable insights for reputation management."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an event impact analyst measuring social media effects on brands and entities.",
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
                    "name": "event_impact_report",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "event_description": {"type": "string"},
                            "affected_entities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "overall_impact_score": {"type": "number"},
                            "reach_estimate": {"type": "integer"},
                            "sentiment_impact": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                            "brand_implications": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "recommended_responses": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "comparison_to_similar_events": {"type": "string"},
                        },
                        "required": [
                            "event_description",
                            "affected_entities",
                            "overall_impact_score",
                            "reach_estimate",
                            "sentiment_impact",
                            "brand_implications",
                            "recommended_responses",
                            "comparison_to_similar_events",
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
        return EventImpactReport.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def predict_event_longevity(event_topic: str) -> EventLongevityPrediction:
    """
    Predict how long an event will remain in discussion.

    Args:
        event_topic: The event to analyze

    Returns:
        EventLongevityPrediction with duration forecast
    """
    # Input validation
    if not event_topic or not event_topic.strip():
        raise ValueError("Event topic cannot be empty")
    event_topic = event_topic.strip()

    prompt = f"""Predict how long "{event_topic}" will remain in discussion on X.

Analyze:
1. Current momentum (building, peak, declining, fading)
2. Predicted total duration of significant discussion
3. Factors that could extend the discussion
4. Factors that could shorten the discussion
5. Comparison to similar past events
6. Best timing for related announcements (if applicable)

This helps plan communications timing."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a social media forecaster predicting the lifespan of trending topics.",
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
                    "name": "event_longevity_prediction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "event_topic": {"type": "string"},
                            "current_momentum": {"type": "string"},
                            "predicted_duration": {"type": "string"},
                            "factors_extending": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "factors_shortening": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "similar_event_comparison": {"type": "string"},
                            "recommended_timing": {"type": "string"},
                        },
                        "required": [
                            "event_topic",
                            "current_momentum",
                            "predicted_duration",
                            "factors_extending",
                            "factors_shortening",
                            "similar_event_comparison",
                            "recommended_timing",
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
        return EventLongevityPrediction.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def get_momentum_color(momentum: str) -> str:
    """Get color for momentum status."""
    return {
        "building": "green",
        "peak": "yellow",
        "declining": "red",
        "fading": "dim",
    }.get(momentum.lower(), "white")


def get_impact_color(score: float) -> str:
    """Get color for impact score."""
    if score >= 7:
        return "red bold"
    elif score >= 5:
        return "yellow"
    return "green"


def main():
    console.print(
        Panel.fit(
            "[bold blue]Event Impact and X Reaction Analysis[/bold blue]\n"
            "Correlate real-world events with social media reactions",
            border_style="blue",
        )
    )

    # Example 1: Analyze event reaction
    console.print("\n[bold yellow]Example 1: Event Reaction Analysis[/bold yellow]")
    console.print("[dim]Analyzing X reaction to a tech announcement[/dim]")

    reaction = analyze_event_reaction(
        event_description="OpenAI GPT-5 announcement",
        hours_since_event=24,
    )

    sentiment = reaction.sentiment_distribution
    mainstream = reaction.mainstream_vs_grassroots

    content = f"""
[bold]Event:[/bold] {reaction.event}
[bold]Total X Mentions:[/bold] {reaction.total_x_mentions:,}
[bold]Peak Hour:[/bold] {reaction.peak_hour}

[bold cyan]Sentiment Distribution:[/bold cyan]
  [green]Positive:[/green] {sentiment.get("positive", 0):.1f}%
  [red]Negative:[/red] {sentiment.get("negative", 0):.1f}%
  [yellow]Neutral:[/yellow] {sentiment.get("neutral", 0):.1f}%

[bold cyan]Source Breakdown:[/bold cyan]
  Mainstream (verified): {mainstream.get("mainstream", 0):.1f}%
  Grassroots (regular): {mainstream.get("grassroots", 0):.1f}%

[bold cyan]Geographic Spread:[/bold cyan]
"""
    for region in reaction.geographic_spread[:4]:
        content += f"  - {region}\n"

    content += "\n[bold cyan]Key Narratives:[/bold cyan]\n"
    for narrative in reaction.key_narratives[:4]:
        content += f"  - {narrative}\n"

    console.print(Panel(content, title="Event Reaction Analysis", border_style="cyan"))

    # Influential reactions
    console.print("\n[bold cyan]Most Influential Reactions:[/bold cyan]")
    for i, reaction_post in enumerate(reaction.most_influential_reactions[:3], 1):
        console.print(f"  {i}. {reaction_post[:80]}...")

    # Example 2: News vs X correlation
    console.print("\n[bold yellow]Example 2: News and X Correlation[/bold yellow]")
    console.print("[dim]Comparing news and X coverage of AI regulation[/dim]")

    correlation = correlate_news_and_social("AI regulation legislation")

    corr_color = {
        "strong": "green",
        "moderate": "yellow",
        "weak": "red",
    }.get(correlation.correlation_strength.lower(), "white")

    content = f"""
[bold]Topic:[/bold] {correlation.topic}
[bold]News Volume:[/bold] {correlation.news_volume}
[bold]X Volume:[/bold] {correlation.x_volume}
[bold]Correlation:[/bold] [{corr_color}]{correlation.correlation_strength.upper()}[/{corr_color}]
[bold]Leading Source:[/bold] {correlation.leading_source}
[bold]Time Lag:[/bold] {correlation.time_lag}

[bold cyan]Key News Outlets:[/bold cyan]
"""
    for outlet in correlation.key_news_outlets[:4]:
        content += f"  - {outlet}\n"

    content += f"""
[bold cyan]X Reaction Summary:[/bold cyan]
{correlation.x_reaction_summary}

[bold cyan]Narrative Differences:[/bold cyan]
"""
    for diff in correlation.narrative_differences[:3]:
        content += f"  - {diff}\n"

    console.print(Panel(content, title="News vs X Correlation", border_style=corr_color))

    # Example 3: Narrative evolution
    console.print("\n[bold yellow]Example 3: Narrative Evolution Tracking[/bold yellow]")
    console.print("[dim]Tracking how narratives evolved for a major event[/dim]")

    evolution = track_narrative_evolution("Apple Vision Pro launch", hours=48)

    content = f"""
[bold]Event:[/bold] {evolution.event_topic}

[bold green]Initial Narrative:[/bold green]
"{evolution.initial_narrative}"

[bold yellow]Dominant Current Narrative:[/bold yellow]
"{evolution.dominant_narrative}"

[bold cyan]Narrative Evolution:[/bold cyan]
"""
    for i, narrative in enumerate(evolution.evolved_narratives[:4], 1):
        content += f"  {i}. {narrative}\n"

    content += "\n[bold red]Counter-Narratives:[/bold red]\n"
    for counter in evolution.counter_narratives[:3]:
        content += f"  - {counter}\n"

    content += "\n[bold cyan]Influencer Impact:[/bold cyan]\n"
    for impact in evolution.influencer_impact[:3]:
        content += f"  - {impact}\n"

    console.print(Panel(content, title="Narrative Evolution", border_style="cyan"))

    # Narrative shift timeline
    if evolution.narrative_shifts:
        shift_table = Table(title="Narrative Shift Timeline", show_header=True)
        shift_table.add_column("Time", style="cyan")
        shift_table.add_column("From")
        shift_table.add_column("To")
        shift_table.add_column("Trigger")

        for shift in evolution.narrative_shifts[:4]:
            shift_table.add_row(
                shift.get("time", "N/A"),
                shift.get("from_narrative", "N/A")[:30] + "...",
                shift.get("to_narrative", "N/A")[:30] + "...",
                shift.get("trigger", "N/A")[:30],
            )

        console.print(shift_table)

    # Example 4: Event impact measurement
    console.print("\n[bold yellow]Example 4: Event Impact Measurement[/bold yellow]")
    console.print("[dim]Measuring impact of a product recall[/dim]")

    impact = measure_event_impact(
        event_keywords=["Tesla", "recall", "autopilot", "safety"],
        affected_entities=["Tesla", "Elon Musk", "NHTSA"],
    )

    impact_color = get_impact_color(impact.overall_impact_score)

    content = f"""
[bold]Event:[/bold] {impact.event_description}
[bold]Impact Score:[/bold] [{impact_color}]{impact.overall_impact_score:.1f}/10[/{impact_color}]
[bold]Estimated Reach:[/bold] {impact.reach_estimate:,}

[bold cyan]Sentiment Impact by Entity:[/bold cyan]
"""
    for entity, change in impact.sentiment_impact.items():
        change_color = "green" if change > 0 else "red" if change < 0 else "yellow"
        sign = "+" if change > 0 else ""
        content += f"  {entity}: [{change_color}]{sign}{change:.2f}[/{change_color}]\n"

    content += "\n[bold cyan]Brand Implications:[/bold cyan]\n"
    for impl in impact.brand_implications[:3]:
        content += f"  - {impl}\n"

    content += f"\n[bold cyan]Comparison:[/bold cyan]\n{impact.comparison_to_similar_events}"

    console.print(Panel(content, title="Event Impact Report", border_style=impact_color.split()[0]))

    console.print("\n[bold green]Recommended Responses:[/bold green]")
    for rec in impact.recommended_responses[:3]:
        console.print(f"  - {rec}")

    # Example 5: Event longevity prediction
    console.print("\n[bold yellow]Example 5: Event Longevity Prediction[/bold yellow]")
    console.print("[dim]Predicting discussion duration[/dim]")

    longevity = predict_event_longevity("AI copyright lawsuit ruling")
    momentum_color = get_momentum_color(longevity.current_momentum)

    content = f"""
[bold]Event:[/bold] {longevity.event_topic}
[bold]Current Momentum:[/bold] [{momentum_color}]{longevity.current_momentum.upper()}[/{momentum_color}]
[bold]Predicted Duration:[/bold] {longevity.predicted_duration}
[bold]Similar Events:[/bold] {longevity.similar_event_comparison}

[bold green]Factors That Could Extend Discussion:[/bold green]
"""
    for factor in longevity.factors_extending[:3]:
        content += f"  - {factor}\n"

    content += "\n[bold red]Factors That Could Shorten Discussion:[/bold red]\n"
    for factor in longevity.factors_shortening[:3]:
        content += f"  - {factor}\n"

    content += f"\n[bold cyan]Recommended Timing:[/bold cyan]\n{longevity.recommended_timing}"

    console.print(Panel(content, title="Longevity Prediction", border_style=momentum_color))

    # Parameter reference
    console.print("\n[bold yellow]X Search Parameters for Event Analysis:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Parameter", style="green")
    ref_table.add_column("Description")
    ref_table.add_column("Use Case")

    ref_table.add_row(
        "from_date / to_date",
        "Time window for analysis",
        "Event timeline tracking",
    )
    ref_table.add_row(
        "sources",
        "Combine X and news",
        '["x", "news"] for correlation',
    )
    ref_table.add_row(
        "post_favorite_count",
        "Filter significant posts",
        "100+ for event reactions",
    )
    ref_table.add_row(
        "max_search_results",
        "Depth of analysis",
        "20 for comprehensive analysis",
    )

    console.print(ref_table)

    console.print("\n[bold cyan]Event Impact Scale:[/bold cyan]")
    scale_table = Table(show_header=True)
    scale_table.add_column("Score", style="bold", justify="center")
    scale_table.add_column("Impact Level")
    scale_table.add_column("Typical Response")

    scale_table.add_row(
        "[green]0-3[/green]",
        "Low - Minor discussion",
        "Monitor, no action needed",
    )
    scale_table.add_row(
        "[yellow]4-6[/yellow]",
        "Moderate - Notable buzz",
        "Active monitoring, consider response",
    )
    scale_table.add_row(
        "[red]7-8[/red]",
        "High - Significant impact",
        "Prepared statement recommended",
    )
    scale_table.add_row(
        "[red bold]9-10[/red bold]",
        "Critical - Major event",
        "Immediate response required",
    )

    console.print(scale_table)


if __name__ == "__main__":
    main()
