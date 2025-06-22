from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path


class VideoProvider(ABC):
    """Abstract base class for video generation providers."""
    
    @abstractmethod
    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 15,
        aspect_ratio: str = "16:9",
        **kwargs
    ) -> Path:
        """Generate video from text prompt."""
        pass


class RunwayMLProvider(VideoProvider):
    """RunwayML video generation provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 15,
        aspect_ratio: str = "16:9",
        **kwargs
    ) -> Path:
        # TODO: Implement RunwayML API integration
        raise NotImplementedError("RunwayML integration pending")


class PikaProvider(VideoProvider):
    """Pika Labs video generation provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 15,
        aspect_ratio: str = "16:9",
        **kwargs
    ) -> Path:
        # TODO: Implement Pika API integration
        raise NotImplementedError("Pika integration pending")


class MockVideoProvider(VideoProvider):
    """Mock video provider for testing."""
    
    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 15,
        aspect_ratio: str = "16:9",
        **kwargs
    ) -> Path:
        """Return mock video path for testing."""
        mock_path = Path(f"outputs/media/mock_video_{hash(prompt)}.mp4")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()  # Create empty file for now
        return mock_path


class VideoFactory:
    """Factory for creating video providers."""
    
    @staticmethod
    def create_provider(provider_type: str, api_key: Optional[str] = None) -> VideoProvider:
        """Create a video provider instance."""
        if provider_type.lower() == "runwayml":
            if not api_key:
                raise ValueError("API key required for RunwayML")
            return RunwayMLProvider(api_key)
        elif provider_type.lower() == "pika":
            if not api_key:
                raise ValueError("API key required for Pika")
            return PikaProvider(api_key)
        elif provider_type.lower() == "mock":
            return MockVideoProvider()
        else:
            raise ValueError(f"Unsupported video provider: {provider_type}")