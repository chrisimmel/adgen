import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel


class Config(BaseModel):
    """Application configuration loaded from YAML and environment variables."""

    # General settings
    ad_duration_seconds: int = 15
    output_dir: str = "outputs"

    # Provider configurations
    providers: Dict[str, str] = {
        "llm": "openai",
        "video": "mock",
        "audio": "mock",
        "music": "mock",
    }

    # Service-specific settings
    llm: Dict[str, Any] = {}
    video: Dict[str, Any] = {}
    audio: Dict[str, Any] = {}
    music: Dict[str, Any] = {}
    review: Dict[str, Any] = {}


def load_config(
    config_path: Optional[Path] = None, env_file: Optional[Path] = None
) -> Config:
    """Load configuration from YAML file and environment variables."""

    # Load environment variables
    if env_file:
        load_dotenv(env_file)
    else:
        # Try to load .env from current directory
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)

    # Load YAML configuration
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    return Config(**config_data)


def get_api_key(provider: str, service_type: str = "llm") -> Optional[str]:
    """Get API key for a specific provider and service type from environment."""

    # Map provider names to environment variable names
    key_mapping = {
        ("openai", "llm"): "OPENAI_API_KEY",
        ("openai", "audio"): "OPENAI_API_KEY",
        ("anthropic", "llm"): "ANTHROPIC_API_KEY",
        ("runwayml", "video"): "RUNWAYML_API_KEY",
        ("pika", "video"): "PIKA_API_KEY",
        ("elevenlabs", "audio"): "ELEVENLABS_API_KEY",
        ("suno", "music"): "SUNO_API_KEY",
        ("udio", "music"): "UDIO_API_KEY",
    }

    env_var = key_mapping.get((provider.lower(), service_type.lower()))
    return os.getenv(env_var) if env_var else None
