# AdGen - AI Video Ad Generator

An agentic system that uses LLMs and generative AI models to create short video advertisements with narration and music.

This is only a small proof of concept, but could easily be the beginning of a full-fledged product.

## Features

- **Agentic Workflow**: Uses LangGraph for orchestrated ad generation
- **Website Analysis**: Automatically analyzes business websites to extract company info
- **Intelligent Business Descriptions**: Generates rich business profiles from web content
- **Video Generation**: Integrated RunwayML and Veo 3 support for AI-generated video content
- **Provider Agnostic**: Abstractions for LLM, video, audio, and music providers
- **Human Review Points**: Strategic approval points in the workflow
- **Markdown Output**: Generated concepts and plans saved as markdown files
- **Flexible Input**: Accepts either website URLs or direct business descriptions
- **CLI Interface**: Clean command line interface with subcommands
- **Workflow Checkpointing**: Automatic checkpoint saving and resumption for cost-effective development
- **Voice-over Generation**: OpenAI TTS integration for narrated video advertisements

## Examples

[Quiksilver with Veo 3](examples/quiksilver-veo3/README.md)

This input:

```bash
uv run adgen generate -u https://www.quiksilver.fr/
```

Yields this ad:

https://github.com/user-attachments/assets/6031deb6-2b99-435a-a60d-bb553ccba67b

## Quick Start

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Generate an ad:**

   **From a website URL:**

   ```bash
   uv run adgen generate -u https://your-business-website.com
   ```

   **From a business description:**

   ```bash
   uv run adgen generate -b "We sell eco-friendly water bottles for athletes"
   ```

   **Interactive mode:**

   ```bash
   uv run adgen generate
   # Follow prompts to choose URL or description input
   ```

## How It Works

AdGen supports two input methods:

### URL Input (Recommended)

1. **Web Scraping**: Extracts content from the business website
2. **Business Analysis**: AI analyzes the content to understand:
   - Company overview and mission
   - Branding style and personality
   - Products and services
   - Target customers and market positioning
3. **Rich Description**: Generates a comprehensive business profile
4. **Ad Creation**: Uses the business profile to create targeted ads

### Direct Description Input

1. **Business Description**: User provides direct business description
2. **Ad Creation**: Immediately proceeds to ad concept generation

Both paths then follow the same workflow:

- **Concept Generation**: Creates ad strategy and messaging
- **Human Review**: Approval checkpoint for concept
- **Script & Visual Planning**: Generates narration and visual elements
- **Video Generation**: Creates AI-generated video using RunwayML or Veo 3 (configurable)
- **Output**: Saves all components as organized markdown files and generated media

## Workflow Checkpointing

AdGen automatically saves checkpoints during workflow execution, allowing you to resume from where you left off if the process is interrupted (e.g., due to API limits, network issues, or system crashes).

### Automatic Checkpointing

Checkpoints are automatically saved after each major workflow step:

- After concept generation
- After script generation
- After visual plan generation
- After video generation
- After audio generation
- After video composition

### Checkpoint Management

**List available checkpoints:**

```bash
uv run adgen checkpoints
```

**Resume from a checkpoint:**

```bash
uv run adgen resume <checkpoint_name>
```

**Delete a checkpoint:**

```bash
uv run adgen delete <checkpoint_name>
```

### Example Checkpoint Workflow

```bash
# Start a new ad generation
uv run adgen generate -u https://example.com

# If interrupted (e.g., ran out of Runway credits), list checkpoints
uv run adgen checkpoints

# Resume from where you left off
uv run adgen resume ad_20231201_143022_concept_generated_20231201_143025
```

Checkpoints are stored in `outputs/checkpoints/` as JSON files containing the complete workflow state.

## CLI Commands

### Generate Commands

```bash
uv run adgen generate -u <url>           # Generate from website URL
uv run adgen generate -b "<description>" # Generate from business description
uv run adgen generate                    # Interactive mode
```

### Checkpoint Management Commands

```bash
uv run adgen checkpoints                 # List all available checkpoints
uv run adgen resume <checkpoint_name>    # Resume from a specific checkpoint
uv run adgen delete <checkpoint_name>    # Delete a checkpoint
```

### Configuration

```bash
# Use custom config file
uv run adgen generate -c custom-config.yaml -u <url>
```

## Configuration

Edit `config.yaml` to customize:

- Provider preferences (OpenAI, Anthropic, Runway, Veo 3, etc.)
- Video duration and quality settings
- Review and approval settings
- Web scraping preferences

## Project Structure

```
adgen/
├── abstractions/     # Provider abstractions (video, audio, music)
├── models/          # Pydantic models for ad components
├── workflows/       # LangGraph workflow definitions
├── utils/           # Configuration and markdown utilities
└── cli.py          # Command line interface
```

## Development Status

Currently implemented:

- ✅ Project structure and configuration
- ✅ Website content extraction and analysis
- ✅ AI-powered business description generation
- ✅ Provider abstractions with mock implementations
- ✅ LangGraph workflow for ad concept generation
- ✅ RunwayML video generation integration
- ✅ Veo 3 video generation integration
- ✅ Video composition with MoviePy
- ✅ OpenAI TTS audio generation integration
- ✅ Voice-over narration with video composition
- ✅ Workflow checkpointing and resumption system
- ✅ CLI interface with subcommands
- ✅ Structured LLM output
- ✅ CLI interface with human review points
- ✅ Markdown output generation

Coming next:

- 🔲 Music generation (Suno, Udio)
- 🔲 ElevenLabs audio generation integration
- 🔲 Automated quality control (the agents will inspect the concept, script, video, etc. to look for things to improve)
- 🔲 FastAPI service interface
