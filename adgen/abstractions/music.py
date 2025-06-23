from abc import ABC, abstractmethod
from pathlib import Path


class MusicProvider(ABC):
    """Abstract base class for AI music generation providers."""

    @abstractmethod
    async def generate_music(
        self,
        prompt: str,
        duration_seconds: int = 15,
        genre: str | None = None,
        mood: str | None = None,
        **kwargs,
    ) -> Path:
        """Generate background music from text prompt."""
        pass


class SunoProvider(MusicProvider):
    """Suno AI music generation provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_music(
        self,
        prompt: str,
        duration_seconds: int = 15,
        genre: str | None = None,
        mood: str | None = None,
        **kwargs,
    ) -> Path:
        # TODO: Implement Suno API integration
        raise NotImplementedError("Suno integration pending")


class UdioProvider(MusicProvider):
    """Udio AI music generation provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_music(
        self,
        prompt: str,
        duration_seconds: int = 15,
        genre: str | None = None,
        mood: str | None = None,
        **kwargs,
    ) -> Path:
        # TODO: Implement Udio API integration
        raise NotImplementedError("Udio integration pending")


class MockMusicProvider(MusicProvider):
    """Mock music provider for testing."""

    async def generate_music(
        self,
        prompt: str,
        _duration_seconds: int = 15,
        _genre: str | None = None,
        _mood: str | None = None,
        **_kwargs,
    ) -> Path:
        """Return mock music path for testing."""
        mock_path = Path(f"outputs/media/mock_music_{hash(prompt)}.mp3")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()  # Create empty file for now
        return mock_path


class MusicFactory:
    """Factory for creating music providers."""

    @staticmethod
    def create_provider(
        provider_type: str, api_key: str | None = None
    ) -> MusicProvider:
        """Create a music provider instance."""
        if provider_type.lower() == "suno":
            if not api_key:
                raise ValueError("API key required for Suno")
            return SunoProvider(api_key)
        elif provider_type.lower() == "udio":
            if not api_key:
                raise ValueError("API key required for Udio")
            return UdioProvider(api_key)
        elif provider_type.lower() == "mock":
            return MockMusicProvider()
        else:
            raise ValueError(f"Unsupported music provider: {provider_type}")
