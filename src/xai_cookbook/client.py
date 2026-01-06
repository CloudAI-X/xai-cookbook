"""xAI API client wrapper using OpenAI SDK."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# Load environment variables
load_dotenv()

XAI_BASE_URL = "https://api.x.ai/v1"


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Get a cached synchronous xAI client.

    Returns:
        OpenAI: Configured client for xAI API

    Raises:
        ValueError: If X_AI_API_KEY is not set
    """
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        raise ValueError("X_AI_API_KEY environment variable is not set")

    return OpenAI(
        api_key=api_key,
        base_url=XAI_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_async_client() -> AsyncOpenAI:
    """Get a cached asynchronous xAI client.

    Returns:
        AsyncOpenAI: Configured async client for xAI API

    Raises:
        ValueError: If X_AI_API_KEY is not set
    """
    api_key = os.environ.get("X_AI_API_KEY")
    if not api_key:
        raise ValueError("X_AI_API_KEY environment variable is not set")

    return AsyncOpenAI(
        api_key=api_key,
        base_url=XAI_BASE_URL,
    )


# Available models with their key characteristics
MODELS = {
    # Grok 4.1 Fast (Latest)
    "grok-4-1-fast-reasoning": {
        "context": 2_000_000,
        "vision": True,
        "reasoning": True,
        "tools": True,
        "input_price": 0.20,
        "output_price": 0.50,
    },
    "grok-4-1-fast-non-reasoning": {
        "context": 2_000_000,
        "vision": True,
        "reasoning": False,
        "tools": True,
        "input_price": 0.20,
        "output_price": 0.50,
    },
    # Grok 4 Fast
    "grok-4-fast-reasoning": {
        "context": 2_000_000,
        "vision": True,
        "reasoning": True,
        "tools": True,
        "input_price": 0.20,
        "output_price": 0.50,
    },
    "grok-4-fast-non-reasoning": {
        "context": 2_000_000,
        "vision": True,
        "reasoning": False,
        "tools": True,
        "input_price": 0.20,
        "output_price": 0.50,
    },
    # Grok 4 Flagship
    "grok-4-0709": {
        "context": 256_000,
        "vision": True,
        "reasoning": False,
        "tools": True,
        "input_price": 3.00,
        "output_price": 15.00,
        "aliases": ["grok-4", "grok-4-latest"],
    },
    # Grok Code
    "grok-code-fast-1": {
        "context": 256_000,
        "vision": True,
        "reasoning": False,
        "tools": True,
        "input_price": 0.20,
        "output_price": 1.50,
        "aliases": ["grok-code-fast"],
    },
    # Grok 3
    "grok-3": {
        "context": 131_072,
        "vision": True,
        "reasoning": False,
        "tools": True,
        "input_price": 3.00,
        "output_price": 15.00,
        "aliases": ["grok-3-latest", "grok-3-beta"],
    },
    "grok-3-mini": {
        "context": 131_072,
        "vision": True,
        "reasoning": False,
        "tools": True,
        "input_price": 0.30,
        "output_price": 0.50,
        "aliases": ["grok-3-mini-latest", "grok-3-mini-beta"],
    },
    # Grok 2
    "grok-2-vision-1212": {
        "context": 32_768,
        "vision": True,
        "reasoning": False,
        "tools": True,
        "input_price": 2.00,
        "output_price": 10.00,
        "aliases": ["grok-2-vision", "grok-2-vision-latest"],
    },
    # Image Generation
    "grok-2-image-1212": {
        "type": "image-generation",
        "price_per_image": 0.07,
        "aliases": ["grok-2-image", "grok-2-image-latest"],
    },
}

# Default model for examples
DEFAULT_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_VISION_MODEL = "grok-4-1-fast-non-reasoning"
DEFAULT_NON_REASONING_MODEL = "grok-4-1-fast-non-reasoning"
DEFAULT_REASONING_MODEL = "grok-4-1-fast-reasoning"
DEFAULT_IMAGE_MODEL = "grok-2-image-1212"
DEFAULT_CODE_MODEL = "grok-code-fast-1"
