<div align="center">

# xAI Cookbook

**75+ examples for every xAI API feature**

[![CI](https://github.com/CloudAI-X/xai-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/CloudAI-X/xai-cookbook/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![xAI API](https://img.shields.io/badge/xAI-API-black.svg)](https://docs.x.ai)

_The most comprehensive collection of xAI API examples, tutorials, and best practices._

[Quick Start](#-quick-start) •
[Examples](#-examples) •
[Models](#-available-models) •
[Learning Paths](#-learning-paths) •
[Contributing](CONTRIBUTING.md)

</div>

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/CloudAI-X/xai-cookbook.git
cd xai-cookbook

# Install dependencies with uv
uv sync

# Set your API key (get one at console.x.ai)
cp .env.example .env
# Edit .env and add your X_AI_API_KEY

# Run your first example
uv run python examples/01_getting_started/01_hello_grok.py
```

## 🤖 Available Models

| Model                         | Context | Features                 | Price (in/out per 1M) |
| ----------------------------- | ------- | ------------------------ | --------------------- |
| `grok-4-1-fast-reasoning` ⭐  | 2M      | Vision, Reasoning, Tools | $0.20/$0.50           |
| `grok-4-1-fast-non-reasoning` | 2M      | Vision, Tools            | $0.20/$0.50           |
| `grok-4-0709`                 | 256K    | Flagship Model           | $3.00/$15.00          |
| `grok-code-fast-1`            | 256K    | Code Specialist          | $0.20/$1.50           |
| `grok-3`                      | 131K    | General Purpose          | $3.00/$15.00          |
| `grok-3-mini`                 | 131K    | Cost-Effective           | $0.30/$0.50           |
| `grok-2-image-1212`           | -       | Image Generation         | $0.07/image           |

⭐ **Recommended default**: `grok-4-1-fast-reasoning` - Latest SOTA with best quality-to-cost ratio.

## 📚 Examples

### Getting Started

| Example                                                              | Description               |
| -------------------------------------------------------------------- | ------------------------- |
| [01_hello_grok.py](examples/01_getting_started/01_hello_grok.py)     | Your first API call       |
| [02_list_models.py](examples/01_getting_started/02_list_models.py)   | List all available models |
| [03_model_info.py](examples/01_getting_started/03_model_info.py)     | Get model details         |
| [04_api_key_info.py](examples/01_getting_started/04_api_key_info.py) | Check API key status      |

### Chat Completions

| Example                                                                               | Description                          |
| ------------------------------------------------------------------------------------- | ------------------------------------ |
| [01_basic_chat.py](examples/02_chat_completions/01_basic_chat.py)                     | Simple chat completion               |
| [02_system_prompts.py](examples/02_chat_completions/02_system_prompts.py)             | Control behavior with system prompts |
| [03_multi_turn.py](examples/02_chat_completions/03_multi_turn.py)                     | Multi-turn conversations             |
| [04_streaming.py](examples/02_chat_completions/04_streaming.py)                       | Stream responses in real-time        |
| [05_async_requests.py](examples/02_chat_completions/05_async_requests.py)             | Async parallel requests              |
| [06_temperature_sampling.py](examples/02_chat_completions/06_temperature_sampling.py) | Temperature and sampling             |
| [07_max_tokens.py](examples/02_chat_completions/07_max_tokens.py)                     | Token limits and stop sequences      |

### Vision

| Example                                                                   | Description                    |
| ------------------------------------------------------------------------- | ------------------------------ |
| [01_image_understanding.py](examples/04_vision/01_image_understanding.py) | Analyze images                 |
| [02_multiple_images.py](examples/04_vision/02_multiple_images.py)         | Multiple images in one request |
| [03_image_and_text.py](examples/04_vision/03_image_and_text.py)           | Mixed content analysis         |
| [04_base64_images.py](examples/04_vision/04_base64_images.py)             | Local images via base64        |
| [05_url_images.py](examples/04_vision/05_url_images.py)                   | Images from URLs               |

### Image Generation

| Example                                                                           | Description       |
| --------------------------------------------------------------------------------- | ----------------- |
| [01_basic_generation.py](examples/05_image_generation/01_basic_generation.py)     | Text-to-image     |
| [02_multiple_images.py](examples/05_image_generation/02_multiple_images.py)       | Batch generation  |
| [03_prompt_engineering.py](examples/05_image_generation/03_prompt_engineering.py) | Effective prompts |
| [04_save_images.py](examples/05_image_generation/04_save_images.py)               | Save to disk      |

### Function Calling

| Example                                                                           | Description          |
| --------------------------------------------------------------------------------- | -------------------- |
| [01_basic_functions.py](examples/06_function_calling/01_basic_functions.py)       | Define functions     |
| [02_function_execution.py](examples/06_function_calling/02_function_execution.py) | Execute and respond  |
| [03_parallel_functions.py](examples/06_function_calling/03_parallel_functions.py) | Parallel calls       |
| [04_complex_schemas.py](examples/06_function_calling/04_complex_schemas.py)       | Nested schemas       |
| [05_real_world_example.py](examples/06_function_calling/05_real_world_example.py) | E-commerce assistant |

### Server-Side Tools

| Example                                                                          | Description         |
| -------------------------------------------------------------------------------- | ------------------- |
| [01_web_search.py](examples/07_server_side_tools/01_web_search.py)               | Web search          |
| [02_x_search.py](examples/07_server_side_tools/02_x_search.py)                   | X/Twitter search    |
| [03_code_execution.py](examples/07_server_side_tools/03_code_execution.py)       | Code execution      |
| [04_document_search.py](examples/07_server_side_tools/04_document_search.py)     | Document search     |
| [05_live_search.py](examples/07_server_side_tools/05_live_search.py)             | Multi-source search |
| [06_combined_tools.py](examples/07_server_side_tools/06_combined_tools.py)       | Multiple tools      |
| [07_tool_results.py](examples/07_server_side_tools/07_tool_results.py)           | Handle results      |
| [08_search_parameters.py](examples/07_server_side_tools/08_search_parameters.py) | Advanced parameters |

### Structured Outputs

| Example                                                                       | Description          |
| ----------------------------------------------------------------------------- | -------------------- |
| [01_json_mode.py](examples/08_structured_outputs/01_json_mode.py)             | Basic JSON mode      |
| [02_json_schema.py](examples/08_structured_outputs/02_json_schema.py)         | Schema validation    |
| [03_strict_mode.py](examples/08_structured_outputs/03_strict_mode.py)         | Strict compliance    |
| [04_pydantic_models.py](examples/08_structured_outputs/04_pydantic_models.py) | Pydantic integration |
| [05_complex_schemas.py](examples/08_structured_outputs/05_complex_schemas.py) | Complex structures   |

### Live Search

| Example                                                          | Description      |
| ---------------------------------------------------------------- | ---------------- |
| [01_web_search.py](examples/09_live_search/01_web_search.py)     | Web search       |
| [02_news_search.py](examples/09_live_search/02_news_search.py)   | News search      |
| [03_multi_source.py](examples/09_live_search/03_multi_source.py) | Combined sources |

### X Search Use Cases ⭐ NEW

Real-world applications leveraging xAI's unique X (Twitter) Search capability.

| Example                                                                                         | Description                | When to Use                                                     |
| ----------------------------------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------------- |
| [01_stock_sentiment.py](examples/17_x_search_use_cases/01_stock_sentiment.py)                   | Stock/crypto sentiment     | Trading signals, market research, portfolio monitoring          |
| [02_brand_monitoring.py](examples/17_x_search_use_cases/02_brand_monitoring.py)                 | Brand reputation tracking  | Marketing teams, PR monitoring, customer feedback analysis      |
| [03_trend_detection.py](examples/17_x_search_use_cases/03_trend_detection.py)                   | Viral content discovery    | Content strategy, marketing campaigns, early trend adoption     |
| [04_competitive_intelligence.py](examples/17_x_search_use_cases/04_competitive_intelligence.py) | Competitor monitoring      | Strategy teams, product launches, market positioning            |
| [05_crisis_monitoring.py](examples/17_x_search_use_cases/05_crisis_monitoring.py)               | Real-time crisis detection | PR teams, risk management, reputation protection                |
| [06_influencer_analytics.py](examples/17_x_search_use_cases/06_influencer_analytics.py)         | KOL tracking and analysis  | Influencer marketing, partnership evaluation, campaign ROI      |
| [07_event_correlation.py](examples/17_x_search_use_cases/07_event_correlation.py)               | Event impact analysis      | News reaction tracking, product launch impact, PR effectiveness |

### Files API

| Example                                                          | Description     |
| ---------------------------------------------------------------- | --------------- |
| [01_upload_file.py](examples/10_files/01_upload_file.py)         | Upload files    |
| [02_list_files.py](examples/10_files/02_list_files.py)           | List files      |
| [03_chat_with_files.py](examples/10_files/03_chat_with_files.py) | Chat with files |
| [04_file_management.py](examples/10_files/04_file_management.py) | File operations |

### Collections (Knowledge Base)

| Example                                                                          | Description          |
| -------------------------------------------------------------------------------- | -------------------- |
| [01_create_collection.py](examples/11_collections/01_create_collection.py)       | Create collection    |
| [02_search_collection.py](examples/11_collections/02_search_collection.py)       | Search collection    |
| [03_chat_with_collection.py](examples/11_collections/03_chat_with_collection.py) | RAG with collections |

### Utilities

| Example                                                            | Description         |
| ------------------------------------------------------------------ | ------------------- |
| [01_tokenization.py](examples/13_utilities/01_tokenization.py)     | Tokenize text       |
| [02_token_counting.py](examples/13_utilities/02_token_counting.py) | Count tokens        |
| [03_rate_limiting.py](examples/13_utilities/03_rate_limiting.py)   | Rate limit handling |

### Advanced Patterns

| Example                                                                 | Description       |
| ----------------------------------------------------------------------- | ----------------- |
| [01_error_handling.py](examples/16_advanced/01_error_handling.py)       | Error handling    |
| [02_retries.py](examples/16_advanced/02_retries.py)                     | Retry strategies  |
| [03_cost_optimization.py](examples/16_advanced/03_cost_optimization.py) | Cost optimization |
| [04_best_practices.py](examples/16_advanced/04_best_practices.py)       | Best practices    |

## 🎓 Learning Paths

### Beginner

1. [Hello Grok](examples/01_getting_started/01_hello_grok.py) - Your first API call
2. [Basic Chat](examples/02_chat_completions/01_basic_chat.py) - Chat fundamentals
3. [Streaming](examples/02_chat_completions/04_streaming.py) - Real-time responses
4. [System Prompts](examples/02_chat_completions/02_system_prompts.py) - Control behavior

### Intermediate

1. [Function Calling](examples/06_function_calling/01_basic_functions.py) - Tool use basics
2. [Structured Outputs](examples/08_structured_outputs/01_json_mode.py) - JSON mode
3. [Vision](examples/04_vision/01_image_understanding.py) - Image understanding
4. [Live Search](examples/07_server_side_tools/01_web_search.py) - Web search integration

### Advanced

1. [Complex Functions](examples/06_function_calling/05_real_world_example.py) - E-commerce assistant
2. [RAG Patterns](examples/11_collections/03_chat_with_collection.py) - Knowledge bases
3. [Error Handling](examples/16_advanced/01_error_handling.py) - Production patterns
4. [Best Practices](examples/16_advanced/04_best_practices.py) - Production readiness

## 💻 Basic Usage

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["X_AI_API_KEY"],
    base_url="https://api.x.ai/v1"
)

response = client.chat.completions.create(
    model="grok-4-1-fast-reasoning",
    messages=[
        {"role": "user", "content": "Hello, Grok!"}
    ]
)

print(response.choices[0].message.content)
```

## 🔗 API Endpoints

| Endpoint                      | Description                          |
| ----------------------------- | ------------------------------------ |
| `POST /v1/chat/completions`   | Chat completions (OpenAI compatible) |
| `POST /v1/images/generations` | Image generation                     |
| `GET /v1/models`              | List available models                |
| `POST /v1/files`              | Upload files                         |
| `POST /v1/tokenize-text`      | Tokenize text                        |
| `GET /v1/api-key`             | API key info                         |

## 📖 Resources

- [xAI Documentation](https://docs.x.ai/docs)
- [API Reference](https://docs.x.ai/api)
- [Models & Pricing](https://docs.x.ai/docs/models)
- [Console](https://console.x.ai)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- [Report a Bug](.github/ISSUE_TEMPLATE/bug_report.md)
- [Request a Feature](.github/ISSUE_TEMPLATE/feature_request.md)
- [Request an Example](.github/ISSUE_TEMPLATE/example_request.md)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ by <a href="https://github.com/CloudAI-X">CloudAI-X</a>

[![Follow on X](https://img.shields.io/badge/Follow-@cloudxdev-1DA1F2?style=flat&logo=x&logoColor=white)](https://x.com/cloudxdev)

</div>
