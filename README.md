# AdGen - AI Video Ad Generator

An agentic system that uses LLMs and generative AI models to create short video advertisements with narration and music.

## Features

- **Agentic Workflow**: Uses LangGraph for orchestrated ad generation
- **Provider Agnostic**: Abstractions for LLM, video, audio, and music providers
- **Human Review Points**: Strategic approval points in the workflow
- **Markdown Output**: Generated concepts and plans saved as markdown files
- **CLI Interface**: Easy-to-use command line interface

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

   ```bash
   uv run adgen -b "We sell eco-friendly water bottles for athletes"
   ```

## Configuration

Edit `config.yaml` to customize:

- Provider preferences (OpenAI, Anthropic, etc.)
- Video duration and quality settings
- Review and approval settings

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
- ✅ Provider abstractions with mock implementations  
- ✅ LangGraph workflow for ad concept generation
- ✅ Structured LLM output
- ✅ CLI interface with human review points
- ✅ Markdown output generation

Coming next:

- 🔲 Actual API integrations for video/audio/music generation
- 🔲 Video composition with MoviePy
- 🔲 FastAPI service interface
  