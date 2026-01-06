#!/usr/bin/env python3
"""
02_brand_monitoring.py - Brand Reputation and Mention Tracking

This example demonstrates how to monitor brand mentions, track sentiment,
and identify reputation risks or opportunities using X posts.

Key concepts:
- Brand mention detection and categorization
- Sentiment analysis for brand health
- Competitor mention comparison
- Alert thresholds for reputation events

Use cases:
- Real-time brand health monitoring
- Customer feedback analysis
- Competitive share of voice tracking
- PR crisis early detection
"""

import json
import os
from datetime import datetime, timedelta

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
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
MIN_FAVORITES_VIRAL = 500
MIN_FAVORITES_TRENDING = 100
DEFAULT_DAYS = 7
DEFAULT_ALERT_THRESHOLD = 2.0

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
class BrandMentionReport(BaseModel):
    """Comprehensive brand mention analysis."""

    brand: str = Field(description="Brand name analyzed")
    total_mentions: int = Field(description="Estimated total mentions found")
    sentiment_breakdown: dict[str, int] = Field(
        description="Count of positive, negative, and neutral mentions"
    )
    overall_sentiment_score: float = Field(description="Overall sentiment from -1.0 to 1.0")
    top_positive_mentions: list[str] = Field(description="Summaries of the most positive mentions")
    top_negative_mentions: list[str] = Field(description="Summaries of the most negative mentions")
    key_topics: list[str] = Field(description="Main topics associated with the brand")
    influential_mentioners: list[str] = Field(description="Notable accounts discussing the brand")


class BrandSentimentTimeline(BaseModel):
    """Brand sentiment tracked over time."""

    brand: str = Field(description="Brand name analyzed")
    time_period: str = Field(description="Time period analyzed")
    sentiment_trend: str = Field(description="Overall trend: improving, declining, or stable")
    daily_sentiments: list[dict] = Field(description="Sentiment scores by day")
    significant_events: list[str] = Field(description="Events that impacted sentiment")
    forecast: str = Field(description="Short-term sentiment forecast")


class ShareOfVoiceReport(BaseModel):
    """Competitive share of voice analysis."""

    time_period: str = Field(description="Analysis time period")
    brands_analyzed: list[str] = Field(description="Brands included in analysis")
    share_of_voice: list[dict] = Field(description="Percentage of mentions for each brand")
    sentiment_comparison: list[dict] = Field(description="Sentiment scores for each brand")
    key_differentiators: list[str] = Field(description="What sets each brand apart in discussions")
    competitive_insights: list[str] = Field(description="Actionable competitive intelligence")


class ReputationEvent(BaseModel):
    """A detected reputation event."""

    event_type: str = Field(description="Type: spike_positive, spike_negative, or trending")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    magnitude: float = Field(description="How far from baseline (standard deviations)")
    trigger_topic: str = Field(description="Topic that triggered the event")
    trigger_posts: list[str] = Field(description="Posts that initiated the event")
    recommended_action: str = Field(description="Suggested response action")
    estimated_reach: int = Field(description="Estimated users who saw related posts")


class MentionCategories(BaseModel):
    """Categorized brand mentions."""

    brand: str = Field(description="Brand name analyzed")
    total_analyzed: int = Field(description="Total mentions categorized")
    complaints: list[dict] = Field(description="Customer complaints with themes")
    praise: list[dict] = Field(description="Positive feedback with themes")
    questions: list[dict] = Field(description="Customer questions and inquiries")
    feature_requests: list[dict] = Field(description="Product feature requests")
    category_counts: dict[str, int] = Field(description="Count of mentions per category")


def get_date_range(days: int) -> tuple[str, str]:
    """Calculate date range for analysis."""
    now = datetime.now()
    from_date = now - timedelta(days=days)
    return from_date.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def monitor_brand_mentions(
    brand_name: str,
    aliases: list[str] | None = None,
    exclude_handles: list[str] | None = None,
) -> BrandMentionReport:
    """
    Monitor and analyze brand mentions on X.

    Args:
        brand_name: Primary brand name
        aliases: Alternative names/hashtags
        exclude_handles: Official accounts to exclude

    Returns:
        BrandMentionReport with comprehensive analysis
    """
    # Input validation
    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")

    search_terms = [brand_name]
    if aliases:
        search_terms.extend(aliases)
    terms_str = ", ".join(search_terms)

    # Handle limit warning for exclude_handles
    if exclude_handles and len(exclude_handles) > MAX_X_HANDLES:
        console.print(
            f"[yellow]Warning: Only excluding first {MAX_X_HANDLES} of "
            f"{len(exclude_handles)} handles (API limit)[/yellow]"
        )
        exclude_handles = exclude_handles[:MAX_X_HANDLES]

    prompt = f"""Analyze brand mentions on X for: {terms_str}

Provide a comprehensive brand monitoring report including:
1. Estimated total number of mentions
2. Sentiment breakdown (positive, negative, neutral counts)
3. Overall sentiment score from -1.0 to 1.0
4. Top 3 positive mentions with summaries
5. Top 3 negative mentions with summaries
6. Key topics and themes associated with the brand
7. Notable/influential accounts discussing the brand

Focus on posts with significant engagement for better signal quality.
{"Exclude posts from these official accounts: " + ", ".join(exclude_handles) if exclude_handles else ""}"""

    x_source: dict = {
        "type": "x",
        "post_favorite_count": MIN_FAVORITES_QUALITY,
    }
    if exclude_handles:
        x_source["excluded_x_handles"] = exclude_handles[:MAX_X_HANDLES]

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a brand monitoring analyst specializing in social media reputation tracking.",
                },
                {"role": "user", "content": prompt},
            ],
            extra_body={
                "search_parameters": {
                    "mode": "on",
                    "return_citations": True,
                    "max_search_results": MAX_SEARCH_RESULTS,
                    "sources": [x_source],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "brand_mention_report",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brand": {"type": "string"},
                            "total_mentions": {"type": "integer"},
                            "sentiment_breakdown": {
                                "type": "object",
                                "properties": {
                                    "positive": {"type": "integer"},
                                    "negative": {"type": "integer"},
                                    "neutral": {"type": "integer"},
                                },
                                "required": ["positive", "negative", "neutral"],
                                "additionalProperties": False,
                            },
                            "overall_sentiment_score": {"type": "number"},
                            "top_positive_mentions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "top_negative_mentions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "key_topics": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "influential_mentioners": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "brand",
                            "total_mentions",
                            "sentiment_breakdown",
                            "overall_sentiment_score",
                            "top_positive_mentions",
                            "top_negative_mentions",
                            "key_topics",
                            "influential_mentioners",
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
        return BrandMentionReport.model_validate_json(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def analyze_brand_sentiment_over_time(brand_name: str, days: int = 7) -> BrandSentimentTimeline:
    """
    Track brand sentiment trends over time.

    Args:
        brand_name: Brand to analyze
        days: Number of days to analyze

    Returns:
        BrandSentimentTimeline with trend analysis
    """
    # Input validation
    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    if days <= 0:
        raise ValueError("days must be a positive integer")

    from_date, to_date = get_date_range(days)

    prompt = f"""Analyze how sentiment toward {brand_name} has evolved on X over the past {days} days.

Provide:
1. Overall sentiment trend (improving, declining, or stable)
2. Approximate daily sentiment scores
3. Any significant events that impacted sentiment
4. A short-term forecast based on current trajectory

Look for patterns and inflection points in the brand's social media perception."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a brand sentiment analyst tracking reputation trends over time.",
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
                    "name": "sentiment_timeline",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brand": {"type": "string"},
                            "time_period": {"type": "string"},
                            "sentiment_trend": {"type": "string"},
                            "daily_sentiments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "date": {"type": "string"},
                                        "sentiment": {"type": "number"},
                                        "notable_event": {"type": "string"},
                                    },
                                    "required": ["date", "sentiment", "notable_event"],
                                    "additionalProperties": False,
                                },
                            },
                            "significant_events": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "forecast": {"type": "string"},
                        },
                        "required": [
                            "brand",
                            "time_period",
                            "sentiment_trend",
                            "daily_sentiments",
                            "significant_events",
                            "forecast",
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
        return BrandSentimentTimeline.model_validate_json(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def compare_brand_share_of_voice(brands: list[str]) -> ShareOfVoiceReport:
    """
    Compare mention volume and sentiment across competing brands.

    Args:
        brands: List of brand names to compare

    Returns:
        ShareOfVoiceReport with competitive analysis
    """
    # Input validation
    if not brands:
        raise ValueError("Brands list cannot be empty")

    brands_str = ", ".join(brands)

    prompt = f"""Compare the share of voice on X for these brands: {brands_str}

Analyze and provide:
1. Approximate percentage of total mentions for each brand (share of voice)
2. Sentiment score for each brand
3. What differentiates each brand in public discussions
4. Actionable competitive insights

Consider both volume of mentions and sentiment quality."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a competitive intelligence analyst comparing brand performance on social media.",
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
                            "post_favorite_count": MIN_FAVORITES_QUALITY,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "share_of_voice_report",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "time_period": {"type": "string"},
                            "brands_analyzed": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "share_of_voice": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "brand": {"type": "string"},
                                        "percentage": {"type": "number"},
                                        "estimated_mentions": {"type": "integer"},
                                    },
                                    "required": [
                                        "brand",
                                        "percentage",
                                        "estimated_mentions",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "sentiment_comparison": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "brand": {"type": "string"},
                                        "sentiment_score": {"type": "number"},
                                        "sentiment_label": {"type": "string"},
                                    },
                                    "required": [
                                        "brand",
                                        "sentiment_score",
                                        "sentiment_label",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "key_differentiators": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "competitive_insights": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "time_period",
                            "brands_analyzed",
                            "share_of_voice",
                            "sentiment_comparison",
                            "key_differentiators",
                            "competitive_insights",
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
        return ShareOfVoiceReport.model_validate_json(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def detect_reputation_events(
    brand_name: str, alert_threshold: float = 2.0
) -> list[ReputationEvent]:
    """
    Detect unusual spikes in negative or positive mentions.

    Args:
        brand_name: Brand to monitor
        alert_threshold: Standard deviations from baseline to trigger alert

    Returns:
        List of ReputationEvent objects
    """
    # Input validation
    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    if alert_threshold <= 0:
        raise ValueError("alert_threshold must be a positive number")

    prompt = f"""Analyze {brand_name} on X for any unusual reputation events.

Look for:
1. Unusual spikes in negative mentions (potential PR issues)
2. Unusual spikes in positive mentions (viral moments, successful campaigns)
3. Trending topics related to the brand

For each event detected, provide:
- Event type (spike_positive, spike_negative, trending)
- Severity (low, medium, high, critical)
- The magnitude of the anomaly
- What triggered it
- Sample posts driving the event
- Recommended action
- Estimated reach

Consider an event significant if it represents a {alert_threshold}x deviation from normal patterns."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a reputation event detector, identifying anomalies in brand mentions.",
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
                    "name": "reputation_events",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "event_type": {"type": "string"},
                                        "severity": {"type": "string"},
                                        "magnitude": {"type": "number"},
                                        "trigger_topic": {"type": "string"},
                                        "trigger_posts": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "recommended_action": {"type": "string"},
                                        "estimated_reach": {"type": "integer"},
                                    },
                                    "required": [
                                        "event_type",
                                        "severity",
                                        "magnitude",
                                        "trigger_topic",
                                        "trigger_posts",
                                        "recommended_action",
                                        "estimated_reach",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["events"],
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
        return [ReputationEvent.model_validate(event) for event in data["events"]]
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def categorize_mentions(brand_name: str) -> MentionCategories:
    """
    Categorize mentions: complaints, praise, questions, feature requests.

    Args:
        brand_name: Brand to analyze

    Returns:
        MentionCategories with categorized mentions
    """
    # Input validation
    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")

    prompt = f"""Categorize recent X mentions of {brand_name} into these categories:

1. Complaints - Customer issues and negative experiences
2. Praise - Positive feedback and compliments
3. Questions - Customer inquiries and support requests
4. Feature Requests - Product improvement suggestions

For each category, provide:
- Example mentions with their themes
- Count of mentions in that category

This helps prioritize customer engagement and product development."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer feedback analyst categorizing social media mentions.",
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
                            "post_favorite_count": MIN_FAVORITES_QUALITY,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "mention_categories",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brand": {"type": "string"},
                            "total_analyzed": {"type": "integer"},
                            "complaints": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "theme": {"type": "string"},
                                        "example": {"type": "string"},
                                        "urgency": {"type": "string"},
                                    },
                                    "required": ["theme", "example", "urgency"],
                                    "additionalProperties": False,
                                },
                            },
                            "praise": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "theme": {"type": "string"},
                                        "example": {"type": "string"},
                                    },
                                    "required": ["theme", "example"],
                                    "additionalProperties": False,
                                },
                            },
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "topic": {"type": "string"},
                                        "example": {"type": "string"},
                                    },
                                    "required": ["topic", "example"],
                                    "additionalProperties": False,
                                },
                            },
                            "feature_requests": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "feature": {"type": "string"},
                                        "example": {"type": "string"},
                                        "demand_level": {"type": "string"},
                                    },
                                    "required": ["feature", "example", "demand_level"],
                                    "additionalProperties": False,
                                },
                            },
                            "category_counts": {
                                "type": "object",
                                "properties": {
                                    "complaints": {"type": "integer"},
                                    "praise": {"type": "integer"},
                                    "questions": {"type": "integer"},
                                    "feature_requests": {"type": "integer"},
                                },
                                "required": [
                                    "complaints",
                                    "praise",
                                    "questions",
                                    "feature_requests",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "required": [
                            "brand",
                            "total_analyzed",
                            "complaints",
                            "praise",
                            "questions",
                            "feature_requests",
                            "category_counts",
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
        return MentionCategories.model_validate_json(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def display_brand_report(report: BrandMentionReport) -> None:
    """Display brand mention report in formatted panels."""
    # Determine overall sentiment color
    if report.overall_sentiment_score > 0.2:
        color = "green"
        label = "POSITIVE"
    elif report.overall_sentiment_score < -0.2:
        color = "red"
        label = "NEGATIVE"
    else:
        color = "yellow"
        label = "NEUTRAL"

    breakdown = report.sentiment_breakdown
    total = (
        breakdown.get("positive", 0) + breakdown.get("negative", 0) + breakdown.get("neutral", 0)
    )

    content = f"""
[bold]Brand:[/bold] {report.brand}
[bold]Total Mentions:[/bold] {report.total_mentions:,}
[bold]Overall Sentiment:[/bold] [{color}]{report.overall_sentiment_score:.2f} ({label})[/{color}]

[bold cyan]Sentiment Breakdown:[/bold cyan]
  [green]Positive:[/green] {breakdown.get("positive", 0):,} ({breakdown.get("positive", 0) / max(total, 1) * 100:.1f}%)
  [red]Negative:[/red] {breakdown.get("negative", 0):,} ({breakdown.get("negative", 0) / max(total, 1) * 100:.1f}%)
  [yellow]Neutral:[/yellow] {breakdown.get("neutral", 0):,} ({breakdown.get("neutral", 0) / max(total, 1) * 100:.1f}%)

[bold cyan]Key Topics:[/bold cyan]
"""
    for topic in report.key_topics[:5]:
        content += f"  - {topic}\n"

    content += "\n[bold cyan]Influential Mentioners:[/bold cyan]\n"
    for mentioner in report.influential_mentioners[:5]:
        content += f"  - {mentioner}\n"

    console.print(
        Panel(
            content,
            title=f"Brand Monitoring Report: {report.brand}",
            border_style=color,
        )
    )


def main():
    console.print(
        Panel.fit(
            "[bold blue]Brand Reputation and Mention Tracking[/bold blue]\n"
            "Monitor brand mentions, sentiment, and reputation events on X",
            border_style="blue",
        )
    )

    # Example 1: Basic brand monitoring
    console.print("\n[bold yellow]Example 1: Brand Mention Monitoring[/bold yellow]")
    console.print("[dim]Analyzing mentions for Apple[/dim]")

    report = monitor_brand_mentions(
        brand_name="Apple",
        aliases=["iPhone", "MacBook", "iOS"],
        exclude_handles=["Apple"],
    )
    display_brand_report(report)

    # Display positive and negative mentions
    if report.top_positive_mentions:
        console.print("\n[bold green]Top Positive Mentions:[/bold green]")
        for i, mention in enumerate(report.top_positive_mentions[:3], 1):
            console.print(f"  {i}. {mention[:100]}...")

    if report.top_negative_mentions:
        console.print("\n[bold red]Top Negative Mentions:[/bold red]")
        for i, mention in enumerate(report.top_negative_mentions[:3], 1):
            console.print(f"  {i}. {mention[:100]}...")

    # Example 2: Sentiment over time
    console.print("\n[bold yellow]Example 2: Sentiment Trend Analysis[/bold yellow]")
    console.print("[dim]Tracking Tesla sentiment over 7 days[/dim]")

    timeline = analyze_brand_sentiment_over_time("Tesla", days=7)

    trend_color = {
        "improving": "green",
        "declining": "red",
        "stable": "yellow",
    }.get(timeline.sentiment_trend.lower(), "white")

    content = f"""
[bold]Brand:[/bold] {timeline.brand}
[bold]Period:[/bold] {timeline.time_period}
[bold]Trend:[/bold] [{trend_color}]{timeline.sentiment_trend.upper()}[/{trend_color}]
[bold]Forecast:[/bold] {timeline.forecast}

[bold cyan]Significant Events:[/bold cyan]
"""
    for event in timeline.significant_events[:3]:
        content += f"  - {event}\n"

    console.print(Panel(content, title="Sentiment Timeline", border_style=trend_color))

    # Daily sentiment table
    if timeline.daily_sentiments:
        table = Table(title="Daily Sentiment", show_header=True)
        table.add_column("Date", style="cyan")
        table.add_column("Sentiment", justify="center")
        table.add_column("Notable Event")

        for day in timeline.daily_sentiments[:7]:
            sentiment = day.get("sentiment", 0)
            if sentiment > 0.2:
                sent_str = f"[green]{sentiment:.2f}[/green]"
            elif sentiment < -0.2:
                sent_str = f"[red]{sentiment:.2f}[/red]"
            else:
                sent_str = f"[yellow]{sentiment:.2f}[/yellow]"

            table.add_row(
                day.get("date", "N/A"),
                sent_str,
                day.get("notable_event", "")[:50],
            )

        console.print(table)

    # Example 3: Share of voice comparison
    console.print("\n[bold yellow]Example 3: Share of Voice Comparison[/bold yellow]")
    console.print("[dim]Comparing OpenAI, Anthropic, and xAI[/dim]")

    sov_report = compare_brand_share_of_voice(["OpenAI", "Anthropic", "xAI"])

    # Share of voice table
    sov_table = Table(title="Share of Voice Analysis", show_header=True)
    sov_table.add_column("Brand", style="cyan")
    sov_table.add_column("Share %", justify="center")
    sov_table.add_column("Est. Mentions", justify="center")
    sov_table.add_column("Sentiment", justify="center")

    # Create sentiment lookup
    sentiment_lookup = {s["brand"]: s for s in sov_report.sentiment_comparison}

    for sov in sov_report.share_of_voice:
        brand = sov["brand"]
        sentiment_data = sentiment_lookup.get(brand, {})
        sentiment_score = sentiment_data.get("sentiment_score", 0)

        if sentiment_score > 0.2:
            sent_str = f"[green]{sentiment_score:.2f}[/green]"
        elif sentiment_score < -0.2:
            sent_str = f"[red]{sentiment_score:.2f}[/red]"
        else:
            sent_str = f"[yellow]{sentiment_score:.2f}[/yellow]"

        sov_table.add_row(
            brand,
            f"{sov['percentage']:.1f}%",
            f"{sov['estimated_mentions']:,}",
            sent_str,
        )

    console.print(sov_table)

    console.print("\n[bold cyan]Competitive Insights:[/bold cyan]")
    for insight in sov_report.competitive_insights[:3]:
        console.print(f"  - {insight}")

    # Example 4: Reputation event detection
    console.print("\n[bold yellow]Example 4: Reputation Event Detection[/bold yellow]")
    console.print("[dim]Scanning for reputation events for Microsoft[/dim]")

    events = detect_reputation_events("Microsoft", alert_threshold=2.0)

    if events:
        for event in events[:3]:
            severity_colors = {
                "critical": "red bold",
                "high": "red",
                "medium": "yellow",
                "low": "green",
            }
            color = severity_colors.get(event.severity.lower(), "white")

            content = f"""
[bold]Type:[/bold] {event.event_type}
[bold]Severity:[/bold] [{color}]{event.severity.upper()}[/{color}]
[bold]Magnitude:[/bold] {event.magnitude:.1f}x from baseline
[bold]Trigger:[/bold] {event.trigger_topic}
[bold]Estimated Reach:[/bold] {event.estimated_reach:,}

[bold cyan]Recommended Action:[/bold cyan]
{event.recommended_action}

[bold cyan]Trigger Posts:[/bold cyan]
"""
            for post in event.trigger_posts[:2]:
                content += f"  - {post[:80]}...\n"

            console.print(
                Panel(
                    content,
                    title=f"REPUTATION EVENT: {event.event_type.upper()}",
                    border_style=color.split()[0],
                )
            )
    else:
        console.print("[green]No significant reputation events detected.[/green]")

    # Example 5: Mention categorization
    console.print("\n[bold yellow]Example 5: Mention Categorization[/bold yellow]")
    console.print("[dim]Categorizing Spotify mentions[/dim]")

    categories = categorize_mentions("Spotify")

    # Category summary table
    cat_table = Table(title="Mention Categories", show_header=True)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="center")
    cat_table.add_column("% of Total", justify="center")

    counts = categories.category_counts
    total = sum(counts.values())

    cat_table.add_row(
        "[red]Complaints[/red]",
        str(counts.get("complaints", 0)),
        f"{counts.get('complaints', 0) / max(total, 1) * 100:.1f}%",
    )
    cat_table.add_row(
        "[green]Praise[/green]",
        str(counts.get("praise", 0)),
        f"{counts.get('praise', 0) / max(total, 1) * 100:.1f}%",
    )
    cat_table.add_row(
        "[yellow]Questions[/yellow]",
        str(counts.get("questions", 0)),
        f"{counts.get('questions', 0) / max(total, 1) * 100:.1f}%",
    )
    cat_table.add_row(
        "[blue]Feature Requests[/blue]",
        str(counts.get("feature_requests", 0)),
        f"{counts.get('feature_requests', 0) / max(total, 1) * 100:.1f}%",
    )

    console.print(cat_table)

    # Show sample from each category
    if categories.complaints:
        console.print("\n[bold red]Top Complaints:[/bold red]")
        for complaint in categories.complaints[:2]:
            console.print(
                f"  [{complaint.get('urgency', 'medium')}] {complaint.get('theme', 'N/A')}: "
                f"{complaint.get('example', '')[:60]}..."
            )

    if categories.feature_requests:
        console.print("\n[bold blue]Top Feature Requests:[/bold blue]")
        for req in categories.feature_requests[:2]:
            console.print(
                f"  [{req.get('demand_level', 'medium')}] {req.get('feature', 'N/A')}: "
                f"{req.get('example', '')[:60]}..."
            )

    # Parameter reference
    console.print("\n[bold yellow]X Search Parameters for Brand Monitoring:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Parameter", style="green")
    ref_table.add_column("Description")
    ref_table.add_column("Use Case")

    ref_table.add_row(
        "excluded_x_handles",
        "Exclude official brand accounts",
        "Filter out brand's own posts",
    )
    ref_table.add_row(
        "post_favorite_count",
        "Minimum likes for quality filtering",
        "Focus on impactful mentions",
    )
    ref_table.add_row(
        "from_date / to_date",
        "Date range for analysis",
        "Trend analysis over time",
    )

    console.print(ref_table)


if __name__ == "__main__":
    main()
