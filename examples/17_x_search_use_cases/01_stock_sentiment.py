#!/usr/bin/env python3
"""
01_stock_sentiment.py - Real-Time Stock/Crypto Sentiment Analysis

This example demonstrates how to analyze social sentiment for financial
instruments using X posts from financial influencers and retail traders.

Key concepts:
- Financial sentiment extraction from X posts
- Ticker symbol monitoring ($AAPL, $BTC)
- Combining X sentiment with engagement metrics
- Structured output for sentiment scores

Use cases:
- Retail investor sentiment tracking
- Pre-market/after-hours sentiment analysis
- Crypto market mood monitoring
- Earnings reaction analysis
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
MIN_FAVORITES_TRENDING = 1000
DEFAULT_TIMEFRAME = "24h"
VALID_TIMEFRAMES = ["1h", "24h", "7d", "30d"]

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
class SentimentResult(BaseModel):
    """Structured result for stock sentiment analysis."""

    ticker: str = Field(description="Stock or crypto ticker symbol")
    sentiment_score: float = Field(
        description="Overall sentiment score from -1.0 (bearish) to 1.0 (bullish)"
    )
    sentiment_label: str = Field(
        description="Sentiment classification: bullish, bearish, or neutral"
    )
    confidence: float = Field(description="Confidence level of the analysis from 0.0 to 1.0")
    post_volume: int = Field(description="Estimated number of posts analyzed")
    top_bullish_posts: list[str] = Field(
        description="Summaries of the most influential bullish posts"
    )
    top_bearish_posts: list[str] = Field(
        description="Summaries of the most influential bearish posts"
    )
    key_themes: list[str] = Field(description="Main themes driving the sentiment")


class InfluencerSentiment(BaseModel):
    """Sentiment from specific financial influencers."""

    topic: str = Field(description="Topic being analyzed")
    influencers_analyzed: list[str] = Field(description="List of influencer handles analyzed")
    overall_sentiment: str = Field(description="Aggregate sentiment: bullish, bearish, or mixed")
    sentiment_by_influencer: list[dict] = Field(
        description="Sentiment breakdown by each influencer"
    )
    consensus_level: str = Field(
        description="How aligned the influencers are: strong, moderate, or weak"
    )
    key_insights: list[str] = Field(description="Notable insights from influencer posts")


class MomentumScore(BaseModel):
    """Sentiment momentum for a ticker."""

    ticker: str = Field(description="Stock or crypto ticker symbol")
    momentum: float = Field(description="Rate of sentiment change, positive means improving")
    direction: str = Field(
        description="Momentum direction: accelerating_bullish, accelerating_bearish, decelerating, or stable"
    )
    current_sentiment: float = Field(description="Current sentiment score")
    previous_sentiment: float = Field(description="Sentiment score from the comparison period")
    volume_change: str = Field(
        description="Change in discussion volume: increasing, decreasing, or stable"
    )


class SentimentShift(BaseModel):
    """Detection of significant sentiment changes."""

    ticker: str = Field(description="Stock or crypto ticker symbol")
    shift_detected: bool = Field(description="Whether a significant shift was detected")
    shift_magnitude: float = Field(description="Size of the sentiment shift from -2.0 to 2.0")
    shift_direction: str = Field(description="Direction of shift: positive, negative, or none")
    trigger_events: list[str] = Field(
        description="Events or news that may have triggered the shift"
    )
    baseline_sentiment: float = Field(description="Baseline sentiment score")
    current_sentiment: float = Field(description="Current sentiment score")
    notable_posts: list[str] = Field(description="Posts that contributed to the shift")


def get_date_range(timeframe: str) -> tuple[str, str]:
    """Calculate date range based on timeframe."""
    if timeframe not in VALID_TIMEFRAMES:
        console.print(
            f"[yellow]Warning: Invalid timeframe '{timeframe}'. "
            f"Using default '{DEFAULT_TIMEFRAME}'.[/yellow]"
        )
        timeframe = DEFAULT_TIMEFRAME

    now = datetime.now()
    if timeframe == "1h":
        from_date = now - timedelta(hours=1)
    elif timeframe == "24h":
        from_date = now - timedelta(days=1)
    elif timeframe == "7d":
        from_date = now - timedelta(days=7)
    elif timeframe == "30d":
        from_date = now - timedelta(days=30)
    else:
        from_date = now - timedelta(days=1)

    return from_date.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def analyze_stock_sentiment(ticker: str, timeframe: str = "24h") -> SentimentResult:
    """
    Analyze X sentiment for a stock/crypto ticker.

    Args:
        ticker: Stock ticker (e.g., "AAPL", "BTC", "TSLA")
        timeframe: "1h", "24h", "7d"

    Returns:
        SentimentResult with score, volume, and key posts
    """
    # Input validation
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol cannot be empty")
    ticker = ticker.strip().upper()

    if timeframe not in VALID_TIMEFRAMES:
        console.print(
            f"[yellow]Warning: Invalid timeframe '{timeframe}'. "
            f"Using default '{DEFAULT_TIMEFRAME}'.[/yellow]"
        )
        timeframe = DEFAULT_TIMEFRAME

    from_date, to_date = get_date_range(timeframe)

    prompt = f"""Analyze the sentiment on X (Twitter) for ${ticker} over the past {timeframe}.

Search for posts mentioning ${ticker}, {ticker}, and related discussions.

Provide a comprehensive sentiment analysis including:
1. An overall sentiment score from -1.0 (extremely bearish) to 1.0 (extremely bullish)
2. Classification as bullish, bearish, or neutral
3. Your confidence level in this analysis
4. Estimated number of relevant posts
5. Summaries of the most influential bullish posts (up to 3)
6. Summaries of the most influential bearish posts (up to 3)
7. Key themes driving the current sentiment

Focus on posts with significant engagement (likes, retweets) for better signal quality."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial sentiment analyst specializing in social media analysis. "
                    "Provide objective, data-driven sentiment assessments based on X posts.",
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
                    "name": "sentiment_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "sentiment_score": {"type": "number"},
                            "sentiment_label": {"type": "string"},
                            "confidence": {"type": "number"},
                            "post_volume": {"type": "integer"},
                            "top_bullish_posts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "top_bearish_posts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "key_themes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "ticker",
                            "sentiment_score",
                            "sentiment_label",
                            "confidence",
                            "post_volume",
                            "top_bullish_posts",
                            "top_bearish_posts",
                            "key_themes",
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
        return SentimentResult.model_validate_json(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def track_financial_influencers(handles: list[str], topic: str) -> InfluencerSentiment:
    """
    Track sentiment from specific financial influencers.

    Args:
        handles: List of X handles (e.g., ["elonmusk", "cathiewood"])
        topic: Topic to filter (e.g., "Tesla", "AI stocks")

    Returns:
        InfluencerSentiment with aggregated insights
    """
    # Input validation
    if not handles:
        raise ValueError("Handles list cannot be empty")
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")

    # Handle limit warning
    if len(handles) > MAX_X_HANDLES:
        console.print(
            f"[yellow]Warning: Only tracking first {MAX_X_HANDLES} of "
            f"{len(handles)} handles (API limit)[/yellow]"
        )
        handles = handles[:MAX_X_HANDLES]

    handles_str = ", ".join([f"@{h}" for h in handles])

    prompt = f"""Analyze what these financial influencers are saying about {topic} on X:
{handles_str}

For each influencer, determine:
1. Their current sentiment toward {topic} (bullish, bearish, or neutral)
2. Key points from their recent posts

Then provide:
- Overall aggregate sentiment across all influencers
- How aligned their views are (consensus level)
- Key insights worth noting for investors"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst tracking key opinion leaders in finance and investing.",
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
                    "name": "influencer_sentiment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "influencers_analyzed": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "overall_sentiment": {"type": "string"},
                            "sentiment_by_influencer": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "handle": {"type": "string"},
                                        "sentiment": {"type": "string"},
                                        "key_point": {"type": "string"},
                                    },
                                    "required": ["handle", "sentiment", "key_point"],
                                    "additionalProperties": False,
                                },
                            },
                            "consensus_level": {"type": "string"},
                            "key_insights": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "topic",
                            "influencers_analyzed",
                            "overall_sentiment",
                            "sentiment_by_influencer",
                            "consensus_level",
                            "key_insights",
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
        return InfluencerSentiment.model_validate_json(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def compare_sentiment_momentum(tickers: list[str]) -> list[MomentumScore]:
    """
    Compare sentiment momentum across multiple tickers.

    Args:
        tickers: List of ticker symbols to compare

    Returns:
        List of MomentumScore for each ticker
    """
    # Input validation
    if not tickers:
        raise ValueError("Tickers list cannot be empty")

    tickers_str = ", ".join([f"${t}" for t in tickers])

    prompt = f"""Compare the sentiment momentum for these tickers on X: {tickers_str}

For each ticker, analyze:
1. Current sentiment score (-1 to 1)
2. How sentiment has changed compared to the previous 24-48 hours
3. The momentum direction (accelerating bullish, accelerating bearish, decelerating, or stable)
4. Whether discussion volume is increasing, decreasing, or stable

Provide a momentum comparison that helps identify which tickers are gaining positive sentiment."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative analyst tracking sentiment momentum across financial assets.",
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
                    "name": "momentum_scores",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "scores": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "ticker": {"type": "string"},
                                        "momentum": {"type": "number"},
                                        "direction": {"type": "string"},
                                        "current_sentiment": {"type": "number"},
                                        "previous_sentiment": {"type": "number"},
                                        "volume_change": {"type": "string"},
                                    },
                                    "required": [
                                        "ticker",
                                        "momentum",
                                        "direction",
                                        "current_sentiment",
                                        "previous_sentiment",
                                        "volume_change",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["scores"],
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
        return [MomentumScore.model_validate(score) for score in data["scores"]]
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def detect_sentiment_shift(ticker: str, baseline_hours: int = 24) -> SentimentShift:
    """
    Detect significant sentiment changes vs baseline.

    Args:
        ticker: Stock or crypto ticker symbol
        baseline_hours: Hours of history for baseline comparison

    Returns:
        SentimentShift with shift detection results
    """
    # Input validation
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol cannot be empty")
    ticker = ticker.strip().upper()

    if baseline_hours <= 0:
        raise ValueError("baseline_hours must be a positive integer")

    prompt = f"""Analyze whether there has been a significant sentiment shift for ${ticker} on X.

Compare the current sentiment (last few hours) against the baseline sentiment from the past {baseline_hours} hours.

Determine:
1. Whether a significant shift has occurred (more than 0.3 change in sentiment score)
2. The magnitude and direction of any shift
3. What events or news might have triggered the shift
4. The baseline and current sentiment scores
5. Notable posts that contributed to any sentiment change

A significant shift could indicate a trading opportunity or risk."""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment shift detector for financial markets, "
                    "identifying sudden changes in market mood.",
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
                    "name": "sentiment_shift",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "shift_detected": {"type": "boolean"},
                            "shift_magnitude": {"type": "number"},
                            "shift_direction": {"type": "string"},
                            "trigger_events": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "baseline_sentiment": {"type": "number"},
                            "current_sentiment": {"type": "number"},
                            "notable_posts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "ticker",
                            "shift_detected",
                            "shift_magnitude",
                            "shift_direction",
                            "trigger_events",
                            "baseline_sentiment",
                            "current_sentiment",
                            "notable_posts",
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
        return SentimentShift.model_validate_json(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse API response: {e}[/red]")
        raise


def display_sentiment_result(result: SentimentResult) -> None:
    """Display sentiment result in a formatted panel."""
    # Determine color based on sentiment
    if result.sentiment_score > 0.2:
        sentiment_color = "green"
        emoji = "[bold green]BULLISH[/bold green]"
    elif result.sentiment_score < -0.2:
        sentiment_color = "red"
        emoji = "[bold red]BEARISH[/bold red]"
    else:
        sentiment_color = "yellow"
        emoji = "[bold yellow]NEUTRAL[/bold yellow]"

    content = f"""
[bold]Ticker:[/bold] ${result.ticker}
[bold]Sentiment Score:[/bold] [{sentiment_color}]{result.sentiment_score:.2f}[/{sentiment_color}] ({emoji})
[bold]Confidence:[/bold] {result.confidence:.0%}
[bold]Post Volume:[/bold] {result.post_volume:,}

[bold cyan]Key Themes:[/bold cyan]
"""
    for theme in result.key_themes[:5]:
        content += f"  - {theme}\n"

    if result.top_bullish_posts:
        content += "\n[bold green]Top Bullish Posts:[/bold green]\n"
        for i, post in enumerate(result.top_bullish_posts[:3], 1):
            content += f"  {i}. {post[:100]}...\n"

    if result.top_bearish_posts:
        content += "\n[bold red]Top Bearish Posts:[/bold red]\n"
        for i, post in enumerate(result.top_bearish_posts[:3], 1):
            content += f"  {i}. {post[:100]}...\n"

    console.print(
        Panel(
            content,
            title=f"Stock Sentiment Analysis: ${result.ticker}",
            border_style=sentiment_color,
        )
    )


def main():
    console.print(
        Panel.fit(
            "[bold blue]Real-Time Stock/Crypto Sentiment Analysis[/bold blue]\n"
            "Analyze social sentiment from X for financial instruments",
            border_style="blue",
        )
    )

    # Example 1: Basic stock sentiment analysis
    console.print("\n[bold yellow]Example 1: Stock Sentiment Analysis[/bold yellow]")
    console.print("[dim]Analyzing sentiment for $TSLA over the past 24 hours[/dim]")

    result = analyze_stock_sentiment("TSLA", "24h")
    display_sentiment_result(result)

    # Example 2: Track financial influencers
    console.print("\n[bold yellow]Example 2: Financial Influencer Tracking[/bold yellow]")
    console.print("[dim]Tracking what key influencers are saying about AI stocks[/dim]")

    influencer_result = track_financial_influencers(
        handles=["elonmusk", "chaikiwood", "jimcramer"],
        topic="AI stocks and technology",
    )

    content = f"""
[bold]Topic:[/bold] {influencer_result.topic}
[bold]Overall Sentiment:[/bold] {influencer_result.overall_sentiment}
[bold]Consensus Level:[/bold] {influencer_result.consensus_level}

[bold cyan]Sentiment by Influencer:[/bold cyan]
"""
    for inf in influencer_result.sentiment_by_influencer:
        content += f"  @{inf['handle']}: {inf['sentiment']} - {inf['key_point'][:80]}...\n"

    content += "\n[bold cyan]Key Insights:[/bold cyan]\n"
    for insight in influencer_result.key_insights[:3]:
        content += f"  - {insight}\n"

    console.print(Panel(content, title="Influencer Sentiment", border_style="magenta"))

    # Example 3: Compare sentiment momentum
    console.print("\n[bold yellow]Example 3: Sentiment Momentum Comparison[/bold yellow]")
    console.print("[dim]Comparing momentum across AAPL, MSFT, GOOGL, NVDA[/dim]")

    momentum_scores = compare_sentiment_momentum(["AAPL", "MSFT", "GOOGL", "NVDA"])

    table = Table(title="Sentiment Momentum Comparison", show_header=True)
    table.add_column("Ticker", style="cyan", justify="center")
    table.add_column("Current", justify="center")
    table.add_column("Previous", justify="center")
    table.add_column("Momentum", justify="center")
    table.add_column("Direction", justify="left")
    table.add_column("Volume", justify="center")

    for score in momentum_scores:
        # Color based on momentum
        if score.momentum > 0:
            momentum_str = f"[green]+{score.momentum:.2f}[/green]"
        elif score.momentum < 0:
            momentum_str = f"[red]{score.momentum:.2f}[/red]"
        else:
            momentum_str = f"[yellow]{score.momentum:.2f}[/yellow]"

        table.add_row(
            f"${score.ticker}",
            f"{score.current_sentiment:.2f}",
            f"{score.previous_sentiment:.2f}",
            momentum_str,
            score.direction,
            score.volume_change,
        )

    console.print(table)

    # Example 4: Detect sentiment shift
    console.print("\n[bold yellow]Example 4: Sentiment Shift Detection[/bold yellow]")
    console.print("[dim]Detecting significant sentiment changes for $BTC[/dim]")

    shift = detect_sentiment_shift("BTC", baseline_hours=24)

    if shift.shift_detected:
        shift_color = "green" if shift.shift_direction == "positive" else "red"
        alert_text = f"[bold {shift_color}]SHIFT DETECTED[/bold {shift_color}]"
    else:
        shift_color = "yellow"
        alert_text = "[bold yellow]NO SIGNIFICANT SHIFT[/bold yellow]"

    content = f"""
{alert_text}

[bold]Baseline Sentiment:[/bold] {shift.baseline_sentiment:.2f}
[bold]Current Sentiment:[/bold] {shift.current_sentiment:.2f}
[bold]Shift Magnitude:[/bold] {shift.shift_magnitude:.2f}
[bold]Direction:[/bold] {shift.shift_direction}

[bold cyan]Trigger Events:[/bold cyan]
"""
    for event in shift.trigger_events[:3]:
        content += f"  - {event}\n"

    if shift.notable_posts:
        content += "\n[bold cyan]Notable Posts:[/bold cyan]\n"
        for post in shift.notable_posts[:3]:
            content += f"  - {post[:80]}...\n"

    console.print(
        Panel(
            content,
            title=f"Sentiment Shift Analysis: ${shift.ticker}",
            border_style=shift_color,
        )
    )

    # Parameter reference
    console.print("\n[bold yellow]X Search Parameters for Financial Analysis:[/bold yellow]")

    ref_table = Table(show_header=True, header_style="bold cyan")
    ref_table.add_column("Parameter", style="green")
    ref_table.add_column("Description")
    ref_table.add_column("Example Value")

    ref_table.add_row(
        "post_favorite_count",
        "Minimum likes to filter quality posts",
        "100 (for significant posts)",
    )
    ref_table.add_row(
        "post_view_count",
        "Minimum views for high-visibility posts",
        "10000",
    )
    ref_table.add_row(
        "included_x_handles",
        "Track specific influencers (max 10)",
        '["elonmusk", "cathiewood"]',
    )
    ref_table.add_row(
        "from_date / to_date",
        "Date range for sentiment analysis",
        '"2024-01-01"',
    )

    console.print(ref_table)


if __name__ == "__main__":
    main()
