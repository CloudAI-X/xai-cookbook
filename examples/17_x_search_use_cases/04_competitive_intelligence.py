#!/usr/bin/env python3
"""
04_competitive_intelligence.py - Competitor Monitoring and Analysis

This example demonstrates how to track competitor activities, announcements,
and public perception through X posts.

Key concepts:
- Competitor mention tracking
- Product launch detection
- Share of voice analysis
- Competitive positioning insights

Use cases:
- Market research and analysis
- Competitive benchmarking
- Product launch monitoring
- Strategic planning support
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
DEFAULT_LOOKBACK_HOURS = 24

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
class CompetitorIntelReport(BaseModel):
    """Comprehensive competitor intelligence report."""

    competitor: str = Field(description="Competitor name")
    mention_volume: int = Field(description="Estimated mention count")
    sentiment_score: float = Field(description="Overall sentiment from -1.0 to 1.0")
    recent_announcements: list[str] = Field(description="Recent announcements or news detected")
    key_themes: list[str] = Field(description="Main themes in discussions about them")
    notable_campaigns: list[str] = Field(description="Marketing campaigns detected")
    threat_level: str = Field(description="Competitive threat level: low, medium, high")
    opportunities: list[str] = Field(
        description="Identified opportunities based on competitor weaknesses"
    )


class CompetitorAnnouncement(BaseModel):
    """A detected competitor announcement."""

    competitor: str = Field(description="Competitor name")
    announcement_type: str = Field(
        description="Type: product, partnership, pricing, hiring, or other"
    )
    summary: str = Field(description="Summary of the announcement")
    source_posts: list[str] = Field(description="Summaries of posts about this announcement")
    market_reaction: str = Field(description="Market/public reaction: positive, negative, or mixed")
    engagement_level: str = Field(description="Engagement level: low, moderate, high, or viral")
    detected_date: str = Field(description="When the announcement was detected")


class PositioningAnalysis(BaseModel):
    """Competitive positioning analysis."""

    your_brand: str = Field(description="Your brand name")
    competitors: list[str] = Field(description="Competitor brands analyzed")
    positioning_map: list[dict] = Field(
        description="How each brand is positioned in public perception"
    )
    differentiation_opportunities: list[str] = Field(
        description="Areas where you can differentiate"
    )
    perception_gaps: list[str] = Field(description="Gaps between desired and actual positioning")
    recommendations: list[str] = Field(description="Strategic recommendations")


class CampaignAnalysis(BaseModel):
    """Analysis of a competitor campaign."""

    campaign_name: str = Field(description="Identified campaign name or theme")
    campaign_type: str = Field(
        description="Type: product launch, brand awareness, promotional, etc."
    )
    key_messages: list[str] = Field(description="Main messages being communicated")
    estimated_reach: int = Field(description="Estimated reach")
    engagement_rate: str = Field(description="Engagement level: low, moderate, high")
    sentiment: str = Field(description="Reception sentiment: positive, negative, mixed")
    lessons_learned: list[str] = Field(description="What can be learned from this campaign")


class ProductSentimentComparison(BaseModel):
    """Comparison of sentiment for competing products."""

    comparison_period: str = Field(description="Time period analyzed")
    products: list[dict] = Field(description="Products with their sentiment data")
    winner: str = Field(description="Product with best sentiment")
    key_differentiators: list[str] = Field(
        description="What differentiates the products in discussions"
    )
    common_complaints: list[dict] = Field(description="Complaints shared across products")
    unique_strengths: list[dict] = Field(description="Unique strengths of each product")


def get_date_range(hours: int) -> tuple[str, str]:
    """Calculate date range for analysis."""
    now = datetime.now()
    from_date = now - timedelta(hours=hours)
    return from_date.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def monitor_competitor(
    competitor_name: str,
    official_handles: list[str],
    product_keywords: list[str],
) -> CompetitorIntelReport:
    """
    Comprehensive competitor monitoring on X.

    Args:
        competitor_name: Company name
        official_handles: Their official X accounts
        product_keywords: Key products/services to track

    Returns:
        CompetitorIntelReport with comprehensive analysis
    """
    # Input validation
    if not competitor_name or not competitor_name.strip():
        raise ValueError("Competitor name cannot be empty")
    competitor_name = competitor_name.strip()

    if not official_handles:
        raise ValueError("Official handles list cannot be empty")

    if not product_keywords:
        raise ValueError("Product keywords list cannot be empty")

    # Handle limit warning
    if len(official_handles) > MAX_X_HANDLES:
        console.print(
            f"[yellow]Warning: Only tracking first {MAX_X_HANDLES} of "
            f"{len(official_handles)} handles (API limit)[/yellow]"
        )
        official_handles = official_handles[:MAX_X_HANDLES]

    keywords_str = ", ".join(product_keywords)
    handles_str = ", ".join([f"@{h}" for h in official_handles])

    prompt = f"""Provide comprehensive competitive intelligence on {competitor_name} from X.

Official accounts: {handles_str}
Products/services to track: {keywords_str}

Analyze and provide:
1. Estimated mention volume
2. Overall sentiment score (-1.0 to 1.0)
3. Recent announcements or news (last 7 days)
4. Key themes in discussions about them
5. Any marketing campaigns detected
6. Competitive threat level (low, medium, high)
7. Opportunities based on their weaknesses or gaps

Focus on actionable competitive intelligence."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a competitive intelligence analyst monitoring market competitors through social media.",
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
                    "name": "competitor_intel_report",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "competitor": {"type": "string"},
                            "mention_volume": {"type": "integer"},
                            "sentiment_score": {"type": "number"},
                            "recent_announcements": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "key_themes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "notable_campaigns": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "threat_level": {"type": "string"},
                            "opportunities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "competitor",
                            "mention_volume",
                            "sentiment_score",
                            "recent_announcements",
                            "key_themes",
                            "notable_campaigns",
                            "threat_level",
                            "opportunities",
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
        return CompetitorIntelReport.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def detect_competitor_announcements(
    competitors: list[str], lookback_hours: int = 24
) -> list[CompetitorAnnouncement]:
    """
    Detect new announcements from competitors.

    Args:
        competitors: List of competitor names to monitor
        lookback_hours: Hours to look back for announcements

    Returns:
        List of CompetitorAnnouncement objects
    """
    # Input validation
    if not competitors:
        raise ValueError("Competitors list cannot be empty")

    # Filter out empty strings
    competitors = [c.strip() for c in competitors if c and c.strip()]
    if not competitors:
        raise ValueError("Competitors list contains only empty strings")

    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be a positive integer")

    competitors_str = ", ".join(competitors)
    from_date, to_date = get_date_range(lookback_hours)

    prompt = f"""Detect recent announcements from these competitors on X: {competitors_str}

Look for:
1. Product launches or updates
2. Partnership announcements
3. Pricing changes
4. Major hiring news
5. Strategic announcements

For each announcement found, provide:
- Competitor name
- Type of announcement
- Summary of the announcement
- Sample posts discussing it
- Market reaction (positive, negative, mixed)
- Engagement level (low, moderate, high, viral)
- When it was detected"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a competitive intelligence analyst detecting competitor announcements from social media.",
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
                    "name": "competitor_announcements",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "announcements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "competitor": {"type": "string"},
                                        "announcement_type": {"type": "string"},
                                        "summary": {"type": "string"},
                                        "source_posts": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "market_reaction": {"type": "string"},
                                        "engagement_level": {"type": "string"},
                                        "detected_date": {"type": "string"},
                                    },
                                    "required": [
                                        "competitor",
                                        "announcement_type",
                                        "summary",
                                        "source_posts",
                                        "market_reaction",
                                        "engagement_level",
                                        "detected_date",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["announcements"],
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
        return [CompetitorAnnouncement.model_validate(a) for a in data["announcements"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def analyze_competitive_positioning(your_brand: str, competitors: list[str]) -> PositioningAnalysis:
    """
    Compare how brands are perceived vs competitors.

    Args:
        your_brand: Your brand name
        competitors: List of competitor brand names

    Returns:
        PositioningAnalysis with strategic insights
    """
    # Input validation
    if not your_brand or not your_brand.strip():
        raise ValueError("Your brand name cannot be empty")
    your_brand = your_brand.strip()

    if not competitors:
        raise ValueError("Competitors list cannot be empty")

    # Filter out empty strings
    competitors = [c.strip() for c in competitors if c and c.strip()]
    if not competitors:
        raise ValueError("Competitors list contains only empty strings")

    all_brands = [your_brand] + competitors
    brands_str = ", ".join(all_brands)

    prompt = f"""Analyze competitive positioning on X for: {brands_str}

Compare how each brand is perceived by the public. For each brand determine:
1. Key positioning attributes (what they're known for)
2. Perceived strengths
3. Perceived weaknesses

Then identify:
- Differentiation opportunities for {your_brand}
- Perception gaps (where actual perception differs from desired)
- Strategic recommendations for improving positioning

Base this on actual X discussions and sentiment."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a brand strategist analyzing competitive positioning through social media perception.",
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
                    "name": "positioning_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "your_brand": {"type": "string"},
                            "competitors": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "positioning_map": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "brand": {"type": "string"},
                                        "key_attributes": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "strengths": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "weaknesses": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "brand",
                                        "key_attributes",
                                        "strengths",
                                        "weaknesses",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "differentiation_opportunities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "perception_gaps": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "recommendations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "your_brand",
                            "competitors",
                            "positioning_map",
                            "differentiation_opportunities",
                            "perception_gaps",
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
        return PositioningAnalysis.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def track_competitor_campaigns(competitor_handle: str) -> list[CampaignAnalysis]:
    """
    Identify and analyze competitor marketing campaigns.

    Args:
        competitor_handle: X handle of the competitor

    Returns:
        List of CampaignAnalysis objects
    """
    # Input validation
    if not competitor_handle or not competitor_handle.strip():
        raise ValueError("Competitor handle cannot be empty")
    competitor_handle = competitor_handle.strip()

    prompt = f"""Analyze marketing campaigns from @{competitor_handle} on X.

Identify active or recent campaigns and for each one provide:
1. Campaign name or theme
2. Campaign type (product launch, brand awareness, promotional, etc.)
3. Key messages being communicated
4. Estimated reach
5. Engagement level (low, moderate, high)
6. Sentiment/reception
7. Lessons that can be learned

Look for coordinated messaging, hashtags, and promotional content."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a marketing analyst studying competitor campaigns on social media.",
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
                            "included_x_handles": [competitor_handle],
                        }
                    ],
                }
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "campaign_analyses",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "campaigns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "campaign_name": {"type": "string"},
                                        "campaign_type": {"type": "string"},
                                        "key_messages": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "estimated_reach": {"type": "integer"},
                                        "engagement_rate": {"type": "string"},
                                        "sentiment": {"type": "string"},
                                        "lessons_learned": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "campaign_name",
                                        "campaign_type",
                                        "key_messages",
                                        "estimated_reach",
                                        "engagement_rate",
                                        "sentiment",
                                        "lessons_learned",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["campaigns"],
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
        return [CampaignAnalysis.model_validate(c) for c in data["campaigns"]]
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def compare_product_sentiment(
    products: dict[str, list[str]],
) -> ProductSentimentComparison:
    """
    Compare sentiment for competing products.

    Args:
        products: Dictionary mapping brand name to product keywords

    Returns:
        ProductSentimentComparison with comparative analysis
    """
    # Input validation
    if not products:
        raise ValueError("Products dictionary cannot be empty")

    # Validate each entry has non-empty brand and keywords
    for brand, keywords in products.items():
        if not brand or not brand.strip():
            raise ValueError("Brand name cannot be empty")
        if not keywords:
            raise ValueError(f"Keywords list for brand '{brand}' cannot be empty")

    products_desc = "; ".join(
        [f"{brand}: {', '.join(keywords)}" for brand, keywords in products.items()]
    )

    prompt = f"""Compare sentiment on X for these competing products:
{products_desc}

For each product/brand, analyze:
1. Sentiment score (-1.0 to 1.0)
2. Volume of discussion
3. Key positive mentions
4. Key negative mentions

Then determine:
- Which product has the best sentiment overall
- Key differentiators in how they're discussed
- Complaints that are common across products
- Unique strengths of each product"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a product analyst comparing competitor sentiment on social media.",
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
                    "name": "product_sentiment_comparison",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "comparison_period": {"type": "string"},
                            "products": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "brand": {"type": "string"},
                                        "product": {"type": "string"},
                                        "sentiment_score": {"type": "number"},
                                        "discussion_volume": {"type": "string"},
                                        "top_positive": {"type": "string"},
                                        "top_negative": {"type": "string"},
                                    },
                                    "required": [
                                        "brand",
                                        "product",
                                        "sentiment_score",
                                        "discussion_volume",
                                        "top_positive",
                                        "top_negative",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "winner": {"type": "string"},
                            "key_differentiators": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "common_complaints": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "complaint": {"type": "string"},
                                        "affected_products": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["complaint", "affected_products"],
                                    "additionalProperties": False,
                                },
                            },
                            "unique_strengths": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "product": {"type": "string"},
                                        "strength": {"type": "string"},
                                    },
                                    "required": ["product", "strength"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "comparison_period",
                            "products",
                            "winner",
                            "key_differentiators",
                            "common_complaints",
                            "unique_strengths",
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
        return ProductSentimentComparison.model_validate_json(response.choices[0].message.content)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Failed to parse response: {e}[/red]")
        raise


def get_threat_color(level: str) -> str:
    """Get color for threat level."""
    return {"low": "green", "medium": "yellow", "high": "red"}.get(level.lower(), "white")


def get_sentiment_color(score: float) -> str:
    """Get color for sentiment score."""
    if score > 0.2:
        return "green"
    elif score < -0.2:
        return "red"
    return "yellow"


def main():
    console.print(
        Panel.fit(
            "[bold blue]Competitor Monitoring and Analysis[/bold blue]\n"
            "Track competitor activities and positioning through X",
            border_style="blue",
        )
    )

    # Example 1: Monitor a competitor
    console.print("\n[bold yellow]Example 1: Comprehensive Competitor Monitoring[/bold yellow]")
    console.print("[dim]Analyzing OpenAI as a competitor[/dim]")

    intel = monitor_competitor(
        competitor_name="OpenAI",
        official_handles=["OpenAI", "sama"],
        product_keywords=["ChatGPT", "GPT-4", "DALL-E", "Sora"],
    )

    threat_color = get_threat_color(intel.threat_level)
    sentiment_color = get_sentiment_color(intel.sentiment_score)

    content = f"""
[bold]Competitor:[/bold] {intel.competitor}
[bold]Mention Volume:[/bold] {intel.mention_volume:,}
[bold]Sentiment:[/bold] [{sentiment_color}]{intel.sentiment_score:.2f}[/{sentiment_color}]
[bold]Threat Level:[/bold] [{threat_color}]{intel.threat_level.upper()}[/{threat_color}]

[bold cyan]Key Themes:[/bold cyan]
"""
    for theme in intel.key_themes[:4]:
        content += f"  - {theme}\n"

    content += "\n[bold cyan]Recent Announcements:[/bold cyan]\n"
    for announcement in intel.recent_announcements[:3]:
        content += f"  - {announcement}\n"

    content += "\n[bold green]Opportunities:[/bold green]\n"
    for opp in intel.opportunities[:3]:
        content += f"  - {opp}\n"

    console.print(
        Panel(
            content,
            title=f"Competitor Intelligence: {intel.competitor}",
            border_style=threat_color,
        )
    )

    # Example 2: Detect announcements
    console.print("\n[bold yellow]Example 2: Competitor Announcement Detection[/bold yellow]")
    console.print("[dim]Detecting announcements from AI companies[/dim]")

    announcements = detect_competitor_announcements(
        competitors=["OpenAI", "Anthropic", "Google AI", "Meta AI"],
        lookback_hours=48,
    )

    if announcements:
        for ann in announcements[:4]:
            reaction_color = {
                "positive": "green",
                "negative": "red",
                "mixed": "yellow",
            }.get(ann.market_reaction.lower(), "white")

            content = f"""
[bold]Type:[/bold] {ann.announcement_type}
[bold]Summary:[/bold] {ann.summary}
[bold]Reaction:[/bold] [{reaction_color}]{ann.market_reaction}[/{reaction_color}]
[bold]Engagement:[/bold] {ann.engagement_level}
[bold]Detected:[/bold] {ann.detected_date}

[bold cyan]Source Posts:[/bold cyan]
"""
            for post in ann.source_posts[:2]:
                content += f"  - {post[:70]}...\n"

            console.print(
                Panel(
                    content,
                    title=f"Announcement: {ann.competitor}",
                    border_style=reaction_color,
                )
            )
    else:
        console.print("[dim]No major announcements detected.[/dim]")

    # Example 3: Competitive positioning
    console.print("\n[bold yellow]Example 3: Competitive Positioning Analysis[/bold yellow]")
    console.print("[dim]Analyzing positioning of xAI vs competitors[/dim]")

    positioning = analyze_competitive_positioning(
        your_brand="xAI", competitors=["OpenAI", "Anthropic"]
    )

    # Positioning map table
    pos_table = Table(title="Competitive Positioning Map", show_header=True)
    pos_table.add_column("Brand", style="cyan")
    pos_table.add_column("Key Attributes")
    pos_table.add_column("Strengths", style="green")
    pos_table.add_column("Weaknesses", style="red")

    for pos in positioning.positioning_map:
        pos_table.add_row(
            pos["brand"],
            ", ".join(pos.get("key_attributes", [])[:2]),
            ", ".join(pos.get("strengths", [])[:2]),
            ", ".join(pos.get("weaknesses", [])[:2]),
        )

    console.print(pos_table)

    console.print("\n[bold green]Differentiation Opportunities:[/bold green]")
    for opp in positioning.differentiation_opportunities[:3]:
        console.print(f"  - {opp}")

    console.print("\n[bold cyan]Strategic Recommendations:[/bold cyan]")
    for rec in positioning.recommendations[:3]:
        console.print(f"  - {rec}")

    # Example 4: Track campaigns
    console.print("\n[bold yellow]Example 4: Competitor Campaign Tracking[/bold yellow]")
    console.print("[dim]Analyzing campaigns from @OpenAI[/dim]")

    campaigns = track_competitor_campaigns("OpenAI")

    if campaigns:
        for campaign in campaigns[:2]:
            engagement_color = {
                "high": "green",
                "moderate": "yellow",
                "low": "red",
            }.get(campaign.engagement_rate.lower(), "white")

            content = f"""
[bold]Type:[/bold] {campaign.campaign_type}
[bold]Estimated Reach:[/bold] {campaign.estimated_reach:,}
[bold]Engagement:[/bold] [{engagement_color}]{campaign.engagement_rate}[/{engagement_color}]
[bold]Sentiment:[/bold] {campaign.sentiment}

[bold cyan]Key Messages:[/bold cyan]
"""
            for msg in campaign.key_messages[:3]:
                content += f"  - {msg}\n"

            content += "\n[bold green]Lessons Learned:[/bold green]\n"
            for lesson in campaign.lessons_learned[:2]:
                content += f"  - {lesson}\n"

            console.print(
                Panel(
                    content,
                    title=f"Campaign: {campaign.campaign_name}",
                    border_style=engagement_color,
                )
            )
    else:
        console.print("[dim]No active campaigns detected.[/dim]")

    # Example 5: Product sentiment comparison
    console.print("\n[bold yellow]Example 5: Product Sentiment Comparison[/bold yellow]")
    console.print("[dim]Comparing AI assistant products[/dim]")

    comparison = compare_product_sentiment(
        {
            "OpenAI": ["ChatGPT", "GPT-4"],
            "Anthropic": ["Claude", "Claude 3"],
            "xAI": ["Grok", "Grok-4"],
        }
    )

    # Product comparison table
    prod_table = Table(title="Product Sentiment Comparison", show_header=True)
    prod_table.add_column("Brand", style="cyan")
    prod_table.add_column("Product")
    prod_table.add_column("Sentiment", justify="center")
    prod_table.add_column("Volume", justify="center")

    for prod in comparison.products:
        sentiment_color = get_sentiment_color(prod["sentiment_score"])
        prod_table.add_row(
            prod["brand"],
            prod["product"],
            f"[{sentiment_color}]{prod['sentiment_score']:.2f}[/{sentiment_color}]",
            prod["discussion_volume"],
        )

    console.print(prod_table)
    console.print(f"\n[bold green]Winner: {comparison.winner}[/bold green]")

    console.print("\n[bold cyan]Key Differentiators:[/bold cyan]")
    for diff in comparison.key_differentiators[:3]:
        console.print(f"  - {diff}")

    console.print("\n[bold cyan]Unique Strengths:[/bold cyan]")
    for strength in comparison.unique_strengths[:4]:
        console.print(f"  - {strength['product']}: {strength['strength']}")

    # Parameter reference
    console.print("\n[bold yellow]X Search Parameters for Competitive Intel:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Parameter", style="green")
    ref_table.add_column("Description")
    ref_table.add_column("Use Case")

    ref_table.add_row(
        "included_x_handles",
        "Track specific competitor accounts",
        "Official account monitoring",
    )
    ref_table.add_row(
        "post_favorite_count",
        "Filter by engagement",
        "Focus on impactful discussions",
    )
    ref_table.add_row(
        "from_date / to_date",
        "Time range for detection",
        "Announcement detection window",
    )
    ref_table.add_row(
        "max_search_results",
        "Number of posts to analyze",
        "Increase for comprehensive analysis",
    )

    console.print(ref_table)


if __name__ == "__main__":
    main()
