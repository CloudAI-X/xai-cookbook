#!/usr/bin/env python3
"""
03_trend_detection.py - Viral Content and Emerging Trend Discovery

This example demonstrates how to identify emerging trends, viral content,
and breaking topics in real-time using X posts.

Key concepts:
- Engagement velocity tracking
- Emerging topic identification
- Viral content characteristics
- Trend lifecycle analysis

Use cases:
- Content marketing timing
- News and media monitoring
- Social listening for product development
- Competitive trend tracking
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
MIN_FAVORITES_VIRAL = 10000
MIN_FAVORITES_TRENDING = 500
DEFAULT_HOURS = 24

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
class EmergingTrend(BaseModel):
    """An emerging trend detected on X."""

    topic: str = Field(description="The trending topic or theme")
    description: str = Field(description="Brief description of the trend")
    engagement_velocity: int = Field(description="Estimated engagements per hour")
    growth_rate: float = Field(description="Percentage growth rate")
    key_posts: list[str] = Field(description="Summaries of key posts driving the trend")
    related_hashtags: list[str] = Field(description="Related hashtags being used")
    predicted_peak: str = Field(
        description="When the trend is expected to peak: hours, days, or already peaked"
    )
    trend_stage: str = Field(description="Current stage: emerging, growing, peak, or declining")


class ViralPost(BaseModel):
    """A viral post analysis."""

    content_summary: str = Field(description="Summary of the post content")
    author: str = Field(description="Author's handle or description")
    likes: int = Field(description="Number of likes/favorites")
    retweets: int = Field(description="Number of retweets")
    views: int = Field(description="Number of views")
    viral_factors: list[str] = Field(description="Factors that contributed to virality")
    content_type: str = Field(description="Type: text, image, video, thread")


class HashtagMomentum(BaseModel):
    """Momentum analysis for a hashtag."""

    hashtag: str = Field(description="The hashtag being tracked")
    current_velocity: int = Field(description="Current posts per hour")
    momentum_direction: str = Field(description="Direction: accelerating, stable, or decelerating")
    peak_time: str = Field(description="When the hashtag peaked or is expected to peak")
    associated_topics: list[str] = Field(description="Topics associated with this hashtag")
    notable_users: list[str] = Field(description="Notable users using this hashtag")


class TrendLifecycle(BaseModel):
    """Lifecycle analysis of a trend."""

    topic: str = Field(description="The trend topic")
    current_stage: str = Field(
        description="Stage: nascent, emerging, mainstream, saturated, or declining"
    )
    estimated_lifespan: str = Field(description="Expected remaining duration of relevance")
    key_milestones: list[dict] = Field(description="Timeline of key moments in the trend")
    recommendation: str = Field(description="Recommendation for engagement timing")


class TrendOriginator(BaseModel):
    """Information about a trend originator."""

    handle: str = Field(description="X handle of the originator")
    role: str = Field(description="Role in the trend: originator, amplifier, or influencer")
    initial_post_summary: str = Field(description="Summary of their contribution")
    follower_count: int = Field(description="Estimated follower count")
    influence_score: float = Field(description="Influence score from 0 to 10")


def get_date_range(hours: int) -> tuple[str, str]:
    """Calculate date range for analysis."""
    now = datetime.now()
    from_date = now - timedelta(hours=hours)
    return from_date.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def discover_emerging_trends(
    domain: str, min_engagement_velocity: int = 100
) -> list[EmergingTrend]:
    """
    Discover trends gaining momentum in a specific domain.

    Args:
        domain: Topic domain to focus on ("tech", "finance", "sports", "general")
        min_engagement_velocity: Minimum engagement per hour to consider

    Returns:
        List of EmergingTrend objects
    """
    # Input validation
    if not domain or not domain.strip():
        raise ValueError("Domain cannot be empty")
    domain = domain.strip()

    if min_engagement_velocity <= 0:
        raise ValueError("min_engagement_velocity must be a positive integer")

    prompt = f"""Discover emerging trends on X in the {domain} domain.

Find topics that are:
1. Gaining momentum rapidly (growing engagement velocity)
2. Have at least {min_engagement_velocity} engagements per hour
3. Are relatively new (emerged in the last 6-24 hours)

For each trend, provide:
- Topic name and description
- Estimated engagement velocity (engagements per hour)
- Growth rate percentage
- Summaries of 2-3 key posts driving the trend
- Related hashtags
- When it's expected to peak
- Current stage (emerging, growing, peak, declining)

Focus on authentic trends, not paid promotions or ads."""

    from_date, to_date = get_date_range(24)

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a trend analyst specializing in identifying emerging viral content and topics on social media.",
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
                    "name": "emerging_trends",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "trends": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "topic": {"type": "string"},
                                        "description": {"type": "string"},
                                        "engagement_velocity": {"type": "integer"},
                                        "growth_rate": {"type": "number"},
                                        "key_posts": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "related_hashtags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "predicted_peak": {"type": "string"},
                                        "trend_stage": {"type": "string"},
                                    },
                                    "required": [
                                        "topic",
                                        "description",
                                        "engagement_velocity",
                                        "growth_rate",
                                        "key_posts",
                                        "related_hashtags",
                                        "predicted_peak",
                                        "trend_stage",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["trends"],
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
        return [EmergingTrend.model_validate(trend) for trend in data["trends"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def analyze_viral_content(topic: str, min_likes: int = 10000) -> list[ViralPost]:
    """
    Find and analyze viral posts on a topic.

    Args:
        topic: Topic to search for viral content
        min_likes: Minimum likes to consider viral

    Returns:
        List of ViralPost objects
    """
    # Input validation
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")
    topic = topic.strip()

    if min_likes <= 0:
        raise ValueError("min_likes must be a positive integer")

    prompt = f"""Find viral posts on X about "{topic}" with high engagement.

Look for posts with at least {min_likes} likes or significant view counts.

For each viral post, analyze:
1. Summary of the content
2. Author (handle or description)
3. Engagement metrics (likes, retweets, views)
4. What factors made it go viral
5. Content type (text, image, video, thread)

Identify patterns in what makes content go viral in this topic area."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a viral content analyst studying what makes posts spread rapidly on X.",
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
                            "post_favorite_count": min_likes,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "viral_posts",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "posts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content_summary": {"type": "string"},
                                        "author": {"type": "string"},
                                        "likes": {"type": "integer"},
                                        "retweets": {"type": "integer"},
                                        "views": {"type": "integer"},
                                        "viral_factors": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "content_type": {"type": "string"},
                                    },
                                    "required": [
                                        "content_summary",
                                        "author",
                                        "likes",
                                        "retweets",
                                        "views",
                                        "viral_factors",
                                        "content_type",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["posts"],
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
        return [ViralPost.model_validate(post) for post in data["posts"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def track_hashtag_momentum(hashtags: list[str]) -> list[HashtagMomentum]:
    """
    Track momentum of specific hashtags.

    Args:
        hashtags: List of hashtags to track (without # symbol)

    Returns:
        List of HashtagMomentum objects
    """
    # Input validation
    if not hashtags:
        raise ValueError("Hashtags list cannot be empty")

    # Filter out empty strings
    hashtags = [h.strip() for h in hashtags if h and h.strip()]
    if not hashtags:
        raise ValueError("Hashtags list contains only empty strings")

    hashtags_str = ", ".join([f"#{h}" for h in hashtags])

    prompt = f"""Analyze the momentum of these hashtags on X: {hashtags_str}

For each hashtag, determine:
1. Current posting velocity (posts per hour)
2. Momentum direction (accelerating, stable, or decelerating)
3. When it peaked or is expected to peak
4. Topics associated with this hashtag
5. Notable users using this hashtag

This helps understand which hashtags are worth engaging with."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a hashtag analyst tracking social media momentum and engagement patterns.",
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
                    "name": "hashtag_momentum",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "hashtags": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "hashtag": {"type": "string"},
                                        "current_velocity": {"type": "integer"},
                                        "momentum_direction": {"type": "string"},
                                        "peak_time": {"type": "string"},
                                        "associated_topics": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "notable_users": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "hashtag",
                                        "current_velocity",
                                        "momentum_direction",
                                        "peak_time",
                                        "associated_topics",
                                        "notable_users",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["hashtags"],
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
        return [HashtagMomentum.model_validate(h) for h in data["hashtags"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def predict_trend_lifecycle(trend_topic: str) -> TrendLifecycle:
    """
    Analyze where a trend is in its lifecycle.

    Args:
        trend_topic: The trend topic to analyze

    Returns:
        TrendLifecycle with stage and predictions
    """
    # Input validation
    if not trend_topic or not trend_topic.strip():
        raise ValueError("Trend topic cannot be empty")
    trend_topic = trend_topic.strip()

    prompt = f"""Analyze the lifecycle stage of the trend "{trend_topic}" on X.

Determine:
1. Current stage (nascent, emerging, mainstream, saturated, or declining)
2. Estimated remaining lifespan of relevance
3. Key milestones in the trend's evolution (with approximate times)
4. Recommendation for whether to engage now or wait

Consider:
- How long the trend has been active
- Current engagement velocity vs peak
- Whether mainstream media has picked it up
- Signs of saturation or fatigue"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a trend lifecycle analyst predicting the trajectory of social media trends.",
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
                            "post_favorite_count": MIN_FAVORITES_TRENDING,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trend_lifecycle",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "current_stage": {"type": "string"},
                            "estimated_lifespan": {"type": "string"},
                            "key_milestones": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "time": {"type": "string"},
                                        "event": {"type": "string"},
                                    },
                                    "required": ["time", "event"],
                                    "additionalProperties": False,
                                },
                            },
                            "recommendation": {"type": "string"},
                        },
                        "required": [
                            "topic",
                            "current_stage",
                            "estimated_lifespan",
                            "key_milestones",
                            "recommendation",
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
        return TrendLifecycle.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def find_trend_originators(trend_topic: str) -> list[TrendOriginator]:
    """
    Identify early adopters who started a trend.

    Args:
        trend_topic: The trend to trace back

    Returns:
        List of TrendOriginator objects
    """
    # Input validation
    if not trend_topic or not trend_topic.strip():
        raise ValueError("Trend topic cannot be empty")
    trend_topic = trend_topic.strip()

    prompt = f"""Identify who started or amplified the trend "{trend_topic}" on X.

Find:
1. Original creators who first posted about this
2. Key amplifiers who helped it go viral
3. Influencers who gave it mainstream attention

For each person, provide:
- Their X handle
- Their role (originator, amplifier, influencer)
- Summary of their contribution
- Estimated follower count
- Influence score (0-10)

This helps understand how trends spread and who drives them."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a trend genealogist tracing the origins and spread of viral topics.",
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
                            "post_favorite_count": MIN_FAVORITES_TRENDING,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trend_originators",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "originators": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "handle": {"type": "string"},
                                        "role": {"type": "string"},
                                        "initial_post_summary": {"type": "string"},
                                        "follower_count": {"type": "integer"},
                                        "influence_score": {"type": "number"},
                                    },
                                    "required": [
                                        "handle",
                                        "role",
                                        "initial_post_summary",
                                        "follower_count",
                                        "influence_score",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["originators"],
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
        return [TrendOriginator.model_validate(o) for o in data["originators"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def get_stage_color(stage: str) -> str:
    """Get color for trend stage."""
    stage_colors = {
        "emerging": "green",
        "nascent": "green",
        "growing": "cyan",
        "peak": "yellow",
        "mainstream": "yellow",
        "saturated": "red",
        "declining": "red",
    }
    return stage_colors.get(stage.lower(), "white")


def get_momentum_color(direction: str) -> str:
    """Get color for momentum direction."""
    if "accelerat" in direction.lower():
        return "green"
    elif "decelerat" in direction.lower() or "declin" in direction.lower():
        return "red"
    return "yellow"


def main():
    console.print(
        Panel.fit(
            "[bold blue]Viral Content and Emerging Trend Discovery[/bold blue]\n"
            "Identify emerging trends and viral content on X in real-time",
            border_style="blue",
        )
    )

    # Example 1: Discover emerging trends
    console.print("\n[bold yellow]Example 1: Emerging Trends in Tech[/bold yellow]")
    console.print("[dim]Discovering trends gaining momentum in the tech domain[/dim]")

    trends = discover_emerging_trends("tech", min_engagement_velocity=100)

    for i, trend in enumerate(trends[:5], 1):
        stage_color = get_stage_color(trend.trend_stage)

        content = f"""
[bold]Description:[/bold] {trend.description}
[bold]Velocity:[/bold] {trend.engagement_velocity:,}/hr ([green]+{trend.growth_rate:.0f}%[/green])
[bold]Stage:[/bold] [{stage_color}]{trend.trend_stage.upper()}[/{stage_color}]
[bold]Predicted Peak:[/bold] {trend.predicted_peak}

[bold cyan]Related Hashtags:[/bold cyan] {", ".join(trend.related_hashtags[:5])}

[bold cyan]Key Posts:[/bold cyan]
"""
        for post in trend.key_posts[:2]:
            content += f"  - {post[:80]}...\n"

        console.print(
            Panel(
                content,
                title=f"Trend #{i}: {trend.topic}",
                border_style=stage_color,
            )
        )

    # Example 2: Analyze viral content
    console.print("\n[bold yellow]Example 2: Viral Content Analysis[/bold yellow]")
    console.print("[dim]Finding viral posts about AI with 10K+ likes[/dim]")

    viral_posts = analyze_viral_content("artificial intelligence", min_likes=10000)

    table = Table(title="Viral Posts Analysis", show_header=True)
    table.add_column("Author", style="cyan")
    table.add_column("Type", justify="center")
    table.add_column("Likes", justify="right")
    table.add_column("Retweets", justify="right")
    table.add_column("Views", justify="right")
    table.add_column("Viral Factors")

    for post in viral_posts[:5]:
        table.add_row(
            post.author[:15],
            post.content_type,
            f"{post.likes:,}",
            f"{post.retweets:,}",
            f"{post.views:,}",
            ", ".join(post.viral_factors[:2]),
        )

    console.print(table)

    # Show content summaries
    console.print("\n[bold cyan]Content Summaries:[/bold cyan]")
    for i, post in enumerate(viral_posts[:3], 1):
        console.print(f"  {i}. {post.content_summary[:100]}...")

    # Example 3: Track hashtag momentum
    console.print("\n[bold yellow]Example 3: Hashtag Momentum Tracking[/bold yellow]")
    console.print("[dim]Tracking momentum of popular hashtags[/dim]")

    hashtags = track_hashtag_momentum(["AI", "GPT", "LLM", "MachineLearning"])

    momentum_table = Table(title="Hashtag Momentum", show_header=True)
    momentum_table.add_column("Hashtag", style="cyan")
    momentum_table.add_column("Velocity", justify="center")
    momentum_table.add_column("Direction", justify="center")
    momentum_table.add_column("Peak", justify="center")
    momentum_table.add_column("Topics")

    for h in hashtags:
        direction_color = get_momentum_color(h.momentum_direction)

        momentum_table.add_row(
            f"#{h.hashtag}",
            f"{h.current_velocity:,}/hr",
            f"[{direction_color}]{h.momentum_direction}[/{direction_color}]",
            h.peak_time,
            ", ".join(h.associated_topics[:2]),
        )

    console.print(momentum_table)

    # Example 4: Trend lifecycle analysis
    console.print("\n[bold yellow]Example 4: Trend Lifecycle Analysis[/bold yellow]")
    console.print("[dim]Analyzing lifecycle of 'Grok' trend[/dim]")

    lifecycle = predict_trend_lifecycle("Grok AI assistant")
    stage_color = get_stage_color(lifecycle.current_stage)

    content = f"""
[bold]Current Stage:[/bold] [{stage_color}]{lifecycle.current_stage.upper()}[/{stage_color}]
[bold]Estimated Lifespan:[/bold] {lifecycle.estimated_lifespan}

[bold cyan]Key Milestones:[/bold cyan]
"""
    for milestone in lifecycle.key_milestones[:5]:
        content += f"  - [{milestone.get('time', 'N/A')}] {milestone.get('event', 'N/A')}\n"

    content += f"""
[bold green]Recommendation:[/bold green]
{lifecycle.recommendation}
"""

    console.print(
        Panel(
            content,
            title=f"Lifecycle Analysis: {lifecycle.topic}",
            border_style=stage_color,
        )
    )

    # Example 5: Find trend originators
    console.print("\n[bold yellow]Example 5: Trend Originators[/bold yellow]")
    console.print("[dim]Identifying who started a viral trend[/dim]")

    originators = find_trend_originators("AI coding assistants")

    originator_table = Table(title="Trend Originators", show_header=True)
    originator_table.add_column("Handle", style="cyan")
    originator_table.add_column("Role", justify="center")
    originator_table.add_column("Followers", justify="right")
    originator_table.add_column("Influence", justify="center")
    originator_table.add_column("Contribution")

    for o in originators[:5]:
        role_color = {
            "originator": "green",
            "amplifier": "yellow",
            "influencer": "cyan",
        }.get(o.role.lower(), "white")

        originator_table.add_row(
            f"@{o.handle}",
            f"[{role_color}]{o.role}[/{role_color}]",
            f"{o.follower_count:,}",
            f"{o.influence_score:.1f}/10",
            o.initial_post_summary[:40] + "...",
        )

    console.print(originator_table)

    # Trend detection tips
    console.print("\n[bold yellow]Trend Detection Parameters:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Parameter", style="green")
    ref_table.add_column("Description")
    ref_table.add_column("Recommended Value")

    ref_table.add_row(
        "post_favorite_count",
        "Filter for quality/viral posts",
        "500+ for trends, 10000+ for viral",
    )
    ref_table.add_row(
        "post_view_count",
        "Filter by visibility",
        "100000+ for high-reach content",
    )
    ref_table.add_row(
        "from_date",
        "Look back period",
        "6-24 hours for emerging trends",
    )
    ref_table.add_row(
        "max_search_results",
        "Number of posts to analyze",
        "15-20 for comprehensive analysis",
    )

    console.print(ref_table)

    console.print("\n[bold cyan]Trend Stage Definitions:[/bold cyan]")
    console.print(
        """
  [green]Nascent/Emerging[/green]: Just starting, high growth potential
  [cyan]Growing[/cyan]: Gaining momentum, not yet mainstream
  [yellow]Peak/Mainstream[/yellow]: Maximum visibility, media coverage
  [red]Saturated/Declining[/red]: Past peak, engagement dropping
    """
    )


if __name__ == "__main__":
    main()
