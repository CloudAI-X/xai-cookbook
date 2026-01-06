#!/usr/bin/env python3
"""
06_influencer_analytics.py - Key Opinion Leader (KOL) Tracking

This example demonstrates how to identify, track, and analyze key influencers
in specific domains using X posts.

Key concepts:
- Influencer discovery and ranking
- Engagement and reach analysis
- Topic authority scoring
- Influencer sentiment toward brands

Use cases:
- Influencer marketing research
- Partnership opportunity discovery
- Brand advocacy identification
- Competitive influencer analysis
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
MIN_FAVORITES_VIRAL = 500
MIN_FOLLOWERS_DEFAULT = 10000
DEFAULT_ANALYSIS_DAYS = 30

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
class Influencer(BaseModel):
    """An identified influencer."""

    handle: str = Field(description="X handle of the influencer")
    name: str = Field(description="Display name")
    follower_count: int = Field(description="Estimated follower count")
    engagement_rate: float = Field(description="Average engagement rate percentage")
    primary_topics: list[str] = Field(description="Main topics they cover")
    average_post_reach: int = Field(description="Average reach per post")
    influence_score: float = Field(description="Influence score from 0 to 10")
    recent_high_performing_posts: list[str] = Field(
        description="Summaries of recent high-performing posts"
    )
    audience_type: str = Field(
        description="Audience type: general, professional, niche, or technical"
    )


class InfluencerAnalysis(BaseModel):
    """Deep analysis of an influencer's content and impact."""

    handle: str = Field(description="X handle")
    analysis_period: str = Field(description="Time period analyzed")
    total_posts: int = Field(description="Posts in the period")
    total_engagement: int = Field(description="Total likes, retweets, replies")
    top_performing_content: list[dict] = Field(description="Top performing posts with metrics")
    content_themes: list[str] = Field(description="Main content themes")
    posting_frequency: str = Field(description="How often they post")
    best_posting_times: list[str] = Field(description="When their posts perform best")
    audience_sentiment: str = Field(
        description="How their audience responds: positive, mixed, or negative"
    )
    brand_mentions: list[dict] = Field(description="Brands they frequently mention")
    collaboration_potential: str = Field(
        description="Potential for partnership: low, medium, or high"
    )


class InfluencerSentiment(BaseModel):
    """Influencer sentiment toward a brand or topic."""

    handle: str = Field(description="X handle")
    sentiment_toward_topic: float = Field(description="Sentiment score from -1.0 to 1.0")
    mention_count: int = Field(description="Number of times mentioned")
    sample_posts: list[str] = Field(description="Sample posts about the topic")
    overall_stance: str = Field(description="Overall stance: advocate, critic, neutral, or mixed")
    recent_sentiment_trend: str = Field(
        description="Trend: becoming more positive, stable, or becoming more negative"
    )


class BrandAdvocate(BaseModel):
    """An organic brand advocate."""

    handle: str = Field(description="X handle")
    name: str = Field(description="Display name")
    mention_count: int = Field(description="Number of brand mentions")
    sentiment_score: float = Field(description="Average sentiment of mentions")
    follower_count: int = Field(description="Estimated followers")
    advocacy_type: str = Field(
        description="Type: organic user, power user, micro-influencer, or creator"
    )
    sample_advocacy: list[str] = Field(description="Examples of advocacy posts")
    engagement_on_brand_content: str = Field(description="Engagement level: low, medium, or high")


class InfluencerComparison(BaseModel):
    """Comparison of multiple influencers."""

    topic: str = Field(description="Topic being compared")
    influencers: list[dict] = Field(description="Influencer comparison data")
    top_performer: str = Field(description="Best performing influencer")
    best_for_reach: str = Field(description="Best for maximum reach")
    best_for_engagement: str = Field(description="Best for engagement rate")
    best_for_brand_fit: str = Field(description="Best fit for brand collaboration")
    recommendations: list[str] = Field(description="Recommendations for partnerships")


def get_date_range(days: int) -> tuple[str, str]:
    """Calculate date range for analysis."""
    now = datetime.now()
    from_date = now - timedelta(days=days)
    return from_date.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def discover_influencers(
    topic: str,
    min_followers: int = 10000,
    min_engagement_rate: float = 0.02,
) -> list[Influencer]:
    """
    Discover influential voices in a topic area.

    Args:
        topic: Topic/niche to search
        min_followers: Minimum follower threshold
        min_engagement_rate: Minimum engagement rate (0.02 = 2%)

    Returns:
        List of Influencer objects
    """
    # Input validation
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")
    topic = topic.strip()

    if min_followers <= 0:
        raise ValueError("min_followers must be a positive integer")

    if min_engagement_rate <= 0:
        raise ValueError("min_engagement_rate must be a positive number")

    prompt = f"""Discover influential voices on X in the topic area: "{topic}"

Find influencers who have:
- At least {min_followers:,} followers
- Strong engagement with their audience
- Authority and credibility in this topic

For each influencer found, provide:
1. Their X handle and display name
2. Estimated follower count
3. Engagement rate (as a percentage)
4. Primary topics they cover
5. Average reach per post
6. Influence score (0-10)
7. Summaries of 2-3 recent high-performing posts
8. Audience type (general, professional, niche, technical)

Focus on genuine influencers, not just accounts with large followings."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an influencer research analyst identifying key opinion leaders on social media.",
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
                            "post_favorite_count": MIN_FAVORITES_VIRAL,
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "influencers",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "influencers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "handle": {"type": "string"},
                                        "name": {"type": "string"},
                                        "follower_count": {"type": "integer"},
                                        "engagement_rate": {"type": "number"},
                                        "primary_topics": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "average_post_reach": {"type": "integer"},
                                        "influence_score": {"type": "number"},
                                        "recent_high_performing_posts": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "audience_type": {"type": "string"},
                                    },
                                    "required": [
                                        "handle",
                                        "name",
                                        "follower_count",
                                        "engagement_rate",
                                        "primary_topics",
                                        "average_post_reach",
                                        "influence_score",
                                        "recent_high_performing_posts",
                                        "audience_type",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["influencers"],
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
        return [Influencer.model_validate(inf) for inf in data["influencers"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def analyze_influencer_content(handle: str, days: int = 30) -> InfluencerAnalysis:
    """
    Deep analysis of an influencer's content and impact.

    Args:
        handle: X handle of the influencer
        days: Number of days to analyze

    Returns:
        InfluencerAnalysis with detailed insights
    """
    # Input validation
    if not handle or not handle.strip():
        raise ValueError("Handle cannot be empty")
    handle = handle.strip()

    if days <= 0:
        raise ValueError("Days must be a positive integer")

    from_date, to_date = get_date_range(days)

    prompt = f"""Analyze the content and impact of @{handle} on X over the past {days} days.

Provide:
1. Total posts in the period
2. Total engagement (likes + retweets + replies)
3. Top 3 performing posts with their metrics
4. Main content themes
5. Posting frequency
6. Best posting times (when their content performs best)
7. Audience sentiment (how their followers respond)
8. Brands they frequently mention (if any)
9. Collaboration potential (low, medium, high) with reasoning

This helps assess their value for partnerships."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an influencer analyst providing deep content analysis for partnership evaluation.",
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
                            "included_x_handles": [handle],
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "influencer_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "handle": {"type": "string"},
                            "analysis_period": {"type": "string"},
                            "total_posts": {"type": "integer"},
                            "total_engagement": {"type": "integer"},
                            "top_performing_content": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "summary": {"type": "string"},
                                        "likes": {"type": "integer"},
                                        "retweets": {"type": "integer"},
                                    },
                                    "required": ["summary", "likes", "retweets"],
                                    "additionalProperties": False,
                                },
                            },
                            "content_themes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "posting_frequency": {"type": "string"},
                            "best_posting_times": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "audience_sentiment": {"type": "string"},
                            "brand_mentions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "brand": {"type": "string"},
                                        "sentiment": {"type": "string"},
                                        "frequency": {"type": "string"},
                                    },
                                    "required": ["brand", "sentiment", "frequency"],
                                    "additionalProperties": False,
                                },
                            },
                            "collaboration_potential": {"type": "string"},
                        },
                        "required": [
                            "handle",
                            "analysis_period",
                            "total_posts",
                            "total_engagement",
                            "top_performing_content",
                            "content_themes",
                            "posting_frequency",
                            "best_posting_times",
                            "audience_sentiment",
                            "brand_mentions",
                            "collaboration_potential",
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
        return InfluencerAnalysis.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def track_influencer_sentiment(
    handles: list[str], brand_or_topic: str
) -> list[InfluencerSentiment]:
    """
    Track what influencers are saying about a brand/topic.

    Args:
        handles: List of X handles to track
        brand_or_topic: Brand or topic to analyze sentiment for

    Returns:
        List of InfluencerSentiment objects
    """
    # Input validation
    if not handles:
        raise ValueError("Handles list cannot be empty")

    # Filter out empty strings
    handles = [h.strip() for h in handles if h and h.strip()]
    if not handles:
        raise ValueError("Handles list contains only empty strings")

    # Handle limit warning
    if len(handles) > MAX_X_HANDLES:
        console.print(
            f"[yellow]Warning: Only tracking first {MAX_X_HANDLES} of "
            f"{len(handles)} handles (API limit)[/yellow]"
        )
        handles = handles[:MAX_X_HANDLES]

    if not brand_or_topic or not brand_or_topic.strip():
        raise ValueError("Brand or topic cannot be empty")
    brand_or_topic = brand_or_topic.strip()

    handles_str = ", ".join([f"@{h}" for h in handles])

    prompt = f"""Analyze what these influencers are saying about "{brand_or_topic}" on X:
{handles_str}

For each influencer, determine:
1. Their sentiment toward {brand_or_topic} (-1.0 to 1.0)
2. How many times they've mentioned it
3. Sample posts about the topic (2-3 examples)
4. Their overall stance (advocate, critic, neutral, mixed)
5. Recent sentiment trend (becoming more positive, stable, becoming more negative)

This helps understand influencer relationships with brands."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an influencer sentiment analyst tracking KOL opinions about brands and topics.",
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
                            "included_x_handles": handles[:MAX_X_HANDLES],
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "influencer_sentiments",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "handle": {"type": "string"},
                                        "sentiment_toward_topic": {"type": "number"},
                                        "mention_count": {"type": "integer"},
                                        "sample_posts": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "overall_stance": {"type": "string"},
                                        "recent_sentiment_trend": {"type": "string"},
                                    },
                                    "required": [
                                        "handle",
                                        "sentiment_toward_topic",
                                        "mention_count",
                                        "sample_posts",
                                        "overall_stance",
                                        "recent_sentiment_trend",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["sentiments"],
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
        return [InfluencerSentiment.model_validate(s) for s in data["sentiments"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def find_brand_advocates(brand_name: str, min_mentions: int = 3) -> list[BrandAdvocate]:
    """
    Find organic brand advocates and fans.

    Args:
        brand_name: Brand to find advocates for
        min_mentions: Minimum positive mentions to qualify

    Returns:
        List of BrandAdvocate objects
    """
    # Input validation
    if not brand_name or not brand_name.strip():
        raise ValueError("Brand name cannot be empty")
    brand_name = brand_name.strip()

    if min_mentions <= 0:
        raise ValueError("min_mentions must be a positive integer")

    prompt = f"""Find organic brand advocates for {brand_name} on X.

Look for users who:
1. Have mentioned {brand_name} positively at least {min_mentions} times
2. Appear to be genuine fans, not paid promoters
3. Have meaningful follower counts

For each advocate found, provide:
- Their X handle and name
- Number of brand mentions
- Average sentiment of their mentions
- Estimated follower count
- Type (organic user, power user, micro-influencer, creator)
- Examples of advocacy posts (2-3)
- Engagement level on brand content

These are potential partners or customer success stories."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a brand advocacy analyst identifying organic supporters and fans.",
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
                    "name": "brand_advocates",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "advocates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "handle": {"type": "string"},
                                        "name": {"type": "string"},
                                        "mention_count": {"type": "integer"},
                                        "sentiment_score": {"type": "number"},
                                        "follower_count": {"type": "integer"},
                                        "advocacy_type": {"type": "string"},
                                        "sample_advocacy": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "engagement_on_brand_content": {"type": "string"},
                                    },
                                    "required": [
                                        "handle",
                                        "name",
                                        "mention_count",
                                        "sentiment_score",
                                        "follower_count",
                                        "advocacy_type",
                                        "sample_advocacy",
                                        "engagement_on_brand_content",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["advocates"],
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
        return [BrandAdvocate.model_validate(a) for a in data["advocates"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def compare_influencer_reach(handles: list[str], topic: str) -> InfluencerComparison:
    """
    Compare reach and impact of multiple influencers.

    Args:
        handles: List of X handles to compare
        topic: Topic context for comparison

    Returns:
        InfluencerComparison with comparative analysis
    """
    # Input validation
    if not handles:
        raise ValueError("Handles list cannot be empty")

    # Filter out empty strings
    handles = [h.strip() for h in handles if h and h.strip()]
    if not handles:
        raise ValueError("Handles list contains only empty strings")

    # Handle limit warning
    if len(handles) > MAX_X_HANDLES:
        console.print(
            f"[yellow]Warning: Only comparing first {MAX_X_HANDLES} of "
            f"{len(handles)} handles (API limit)[/yellow]"
        )
        handles = handles[:MAX_X_HANDLES]

    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")
    topic = topic.strip()

    handles_str = ", ".join([f"@{h}" for h in handles])

    prompt = f"""Compare these influencers for partnerships related to "{topic}":
{handles_str}

For each influencer, compare:
1. Follower count
2. Engagement rate
3. Topic relevance
4. Audience quality
5. Brand safety
6. Content style fit

Then determine:
- Overall top performer
- Best for maximum reach
- Best for engagement rate
- Best for brand fit

Provide recommendations for which to partner with and why."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an influencer partnership advisor comparing KOLs for brand collaborations.",
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
                            "included_x_handles": handles[:MAX_X_HANDLES],
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "influencer_comparison",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "influencers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "handle": {"type": "string"},
                                        "followers": {"type": "integer"},
                                        "engagement_rate": {"type": "number"},
                                        "topic_relevance": {"type": "string"},
                                        "audience_quality": {"type": "string"},
                                        "brand_safety": {"type": "string"},
                                        "overall_score": {"type": "number"},
                                    },
                                    "required": [
                                        "handle",
                                        "followers",
                                        "engagement_rate",
                                        "topic_relevance",
                                        "audience_quality",
                                        "brand_safety",
                                        "overall_score",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "top_performer": {"type": "string"},
                            "best_for_reach": {"type": "string"},
                            "best_for_engagement": {"type": "string"},
                            "best_for_brand_fit": {"type": "string"},
                            "recommendations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "topic",
                            "influencers",
                            "top_performer",
                            "best_for_reach",
                            "best_for_engagement",
                            "best_for_brand_fit",
                            "recommendations",
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
        return InfluencerComparison.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def get_influence_color(score: float) -> str:
    """Get color for influence score."""
    if score >= 8:
        return "green bold"
    elif score >= 6:
        return "green"
    elif score >= 4:
        return "yellow"
    return "red"


def get_sentiment_color(score: float) -> str:
    """Get color for sentiment score."""
    if score > 0.3:
        return "green"
    elif score < -0.3:
        return "red"
    return "yellow"


def main():
    console.print(
        Panel.fit(
            "[bold blue]Key Opinion Leader (KOL) Tracking[/bold blue]\n"
            "Identify, track, and analyze influencers on X",
            border_style="blue",
        )
    )

    # Example 1: Discover influencers
    console.print("\n[bold yellow]Example 1: Influencer Discovery[/bold yellow]")
    console.print("[dim]Discovering AI/ML influencers[/dim]")

    influencers = discover_influencers(
        topic="Artificial Intelligence and Machine Learning",
        min_followers=10000,
        min_engagement_rate=0.02,
    )

    table = Table(title="Discovered Influencers", show_header=True)
    table.add_column("Handle", style="cyan")
    table.add_column("Followers", justify="right")
    table.add_column("Engagement", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Topics")

    for inf in influencers[:6]:
        score_color = get_influence_color(inf.influence_score)

        table.add_row(
            f"@{inf.handle}",
            f"{inf.follower_count:,}",
            f"{inf.engagement_rate:.1%}",
            f"[{score_color}]{inf.influence_score:.1f}[/{score_color}]",
            ", ".join(inf.primary_topics[:2]),
        )

    console.print(table)

    # Show top performer details
    if influencers:
        top = influencers[0]
        console.print(f"\n[bold cyan]Top Performer: @{top.handle}[/bold cyan]")
        console.print(f"  Audience Type: {top.audience_type}")
        console.print("  Recent High-Performing Posts:")
        for post in top.recent_high_performing_posts[:2]:
            console.print(f"    - {post[:80]}...")

    # Example 2: Analyze influencer content
    console.print("\n[bold yellow]Example 2: Influencer Content Analysis[/bold yellow]")
    console.print("[dim]Analyzing content from @elonmusk[/dim]")

    analysis = analyze_influencer_content("elonmusk", days=30)

    collab_color = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
    }.get(analysis.collaboration_potential.lower().split()[0], "white")

    content = f"""
[bold]Handle:[/bold] @{analysis.handle}
[bold]Period:[/bold] {analysis.analysis_period}
[bold]Total Posts:[/bold] {analysis.total_posts}
[bold]Total Engagement:[/bold] {analysis.total_engagement:,}
[bold]Posting Frequency:[/bold] {analysis.posting_frequency}
[bold]Audience Sentiment:[/bold] {analysis.audience_sentiment}
[bold]Collaboration Potential:[/bold] [{collab_color}]{analysis.collaboration_potential}[/{collab_color}]

[bold cyan]Content Themes:[/bold cyan]
"""
    for theme in analysis.content_themes[:4]:
        content += f"  - {theme}\n"

    content += "\n[bold cyan]Best Posting Times:[/bold cyan]\n"
    for time in analysis.best_posting_times[:3]:
        content += f"  - {time}\n"

    console.print(Panel(content, title="Influencer Analysis", border_style="cyan"))

    # Top performing content table
    if analysis.top_performing_content:
        perf_table = Table(title="Top Performing Content", show_header=True)
        perf_table.add_column("Content", max_width=50)
        perf_table.add_column("Likes", justify="right", style="green")
        perf_table.add_column("Retweets", justify="right", style="cyan")

        for post in analysis.top_performing_content[:3]:
            perf_table.add_row(
                post.get("summary", "")[:50] + "...",
                f"{post.get('likes', 0):,}",
                f"{post.get('retweets', 0):,}",
            )

        console.print(perf_table)

    # Example 3: Track influencer sentiment
    console.print("\n[bold yellow]Example 3: Influencer Sentiment Tracking[/bold yellow]")
    console.print("[dim]Tracking influencer sentiment toward OpenAI[/dim]")

    sentiments = track_influencer_sentiment(
        handles=["elonmusk", "sama", "ylecun"],
        brand_or_topic="OpenAI",
    )

    sent_table = Table(title="Influencer Sentiment", show_header=True)
    sent_table.add_column("Handle", style="cyan")
    sent_table.add_column("Sentiment", justify="center")
    sent_table.add_column("Mentions", justify="center")
    sent_table.add_column("Stance", justify="center")
    sent_table.add_column("Trend")

    for sent in sentiments:
        sent_color = get_sentiment_color(sent.sentiment_toward_topic)
        stance_color = {
            "advocate": "green",
            "critic": "red",
            "neutral": "yellow",
            "mixed": "yellow",
        }.get(sent.overall_stance.lower(), "white")

        sent_table.add_row(
            f"@{sent.handle}",
            f"[{sent_color}]{sent.sentiment_toward_topic:.2f}[/{sent_color}]",
            str(sent.mention_count),
            f"[{stance_color}]{sent.overall_stance}[/{stance_color}]",
            sent.recent_sentiment_trend,
        )

    console.print(sent_table)

    # Example 4: Find brand advocates
    console.print("\n[bold yellow]Example 4: Brand Advocate Discovery[/bold yellow]")
    console.print("[dim]Finding organic advocates for Tesla[/dim]")

    advocates = find_brand_advocates("Tesla", min_mentions=3)

    if advocates:
        adv_table = Table(title="Brand Advocates", show_header=True)
        adv_table.add_column("Handle", style="cyan")
        adv_table.add_column("Name")
        adv_table.add_column("Mentions", justify="center")
        adv_table.add_column("Sentiment", justify="center")
        adv_table.add_column("Type")
        adv_table.add_column("Engagement")

        for adv in advocates[:5]:
            sent_color = get_sentiment_color(adv.sentiment_score)

            adv_table.add_row(
                f"@{adv.handle}",
                adv.name[:20],
                str(adv.mention_count),
                f"[{sent_color}]{adv.sentiment_score:.2f}[/{sent_color}]",
                adv.advocacy_type,
                adv.engagement_on_brand_content,
            )

        console.print(adv_table)

        # Show sample advocacy
        if advocates[0].sample_advocacy:
            console.print(f"\n[bold cyan]Sample Advocacy from @{advocates[0].handle}:[/bold cyan]")
            for post in advocates[0].sample_advocacy[:2]:
                console.print(f"  - {post[:80]}...")
    else:
        console.print("[dim]No strong brand advocates found.[/dim]")

    # Example 5: Compare influencers
    console.print("\n[bold yellow]Example 5: Influencer Comparison[/bold yellow]")
    console.print("[dim]Comparing AI influencers for potential partnership[/dim]")

    comparison = compare_influencer_reach(
        handles=["AndrewYNg", "kaborov", "ylecun", "hardmaru"],
        topic="AI/ML education and research",
    )

    comp_table = Table(title="Influencer Comparison", show_header=True)
    comp_table.add_column("Handle", style="cyan")
    comp_table.add_column("Followers", justify="right")
    comp_table.add_column("Engagement", justify="center")
    comp_table.add_column("Relevance", justify="center")
    comp_table.add_column("Safety", justify="center")
    comp_table.add_column("Score", justify="center")

    for inf in comparison.influencers:
        score_color = get_influence_color(inf.get("overall_score", 0))

        comp_table.add_row(
            f"@{inf['handle']}",
            f"{inf.get('followers', 0):,}",
            f"{inf.get('engagement_rate', 0):.2%}",
            inf.get("topic_relevance", "N/A"),
            inf.get("brand_safety", "N/A"),
            f"[{score_color}]{inf.get('overall_score', 0):.1f}[/{score_color}]",
        )

    console.print(comp_table)

    console.print("\n[bold green]Winners:[/bold green]")
    console.print(f"  Top Performer: @{comparison.top_performer}")
    console.print(f"  Best for Reach: @{comparison.best_for_reach}")
    console.print(f"  Best for Engagement: @{comparison.best_for_engagement}")
    console.print(f"  Best for Brand Fit: @{comparison.best_for_brand_fit}")

    console.print("\n[bold cyan]Recommendations:[/bold cyan]")
    for rec in comparison.recommendations[:3]:
        console.print(f"  - {rec}")

    # Parameter reference
    console.print("\n[bold yellow]X Search Parameters for Influencer Analysis:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Parameter", style="green")
    ref_table.add_column("Description")
    ref_table.add_column("Use Case")

    ref_table.add_row(
        "included_x_handles",
        "Track specific influencers",
        "Content analysis, sentiment tracking",
    )
    ref_table.add_row(
        "post_favorite_count",
        "Filter high-engagement posts",
        "500+ for influencer discovery",
    )
    ref_table.add_row(
        "from_date / to_date",
        "Analysis time window",
        "30 days for content analysis",
    )

    console.print(ref_table)

    console.print("\n[bold cyan]Influencer Tiers:[/bold cyan]")
    tier_table = Table(show_header=True)
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Followers")
    tier_table.add_column("Typical Engagement")
    tier_table.add_column("Best For")

    tier_table.add_row("Nano", "1K-10K", "5-10%", "Niche communities, authenticity")
    tier_table.add_row("Micro", "10K-100K", "3-5%", "Targeted reach, strong engagement")
    tier_table.add_row("Mid-tier", "100K-500K", "2-3%", "Balanced reach and engagement")
    tier_table.add_row("Macro", "500K-1M", "1-2%", "Mass awareness, brand visibility")
    tier_table.add_row("Mega", "1M+", "<1%", "Maximum reach, celebrity status")

    console.print(tier_table)


if __name__ == "__main__":
    main()
