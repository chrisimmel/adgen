from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path


class AudioProvider(ABC):
    """Abstract base class for text-to-speech providers."""
    
    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> Path:
        """Generate speech audio from text."""
        pass


class ElevenLabsProvider(AudioProvider):
    """ElevenLabs text-to-speech provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> Path:
        # TODO: Implement ElevenLabs API integration
        raise NotImplementedError("ElevenLabs integration pending")


class OpenAITTSProvider(AudioProvider):
    """OpenAI text-to-speech provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = "alloy",
        speed: float = 1.0,
        **kwargs
    ) -> Path:
        # TODO: Implement OpenAI TTS API integration
        raise NotImplementedError("OpenAI TTS integration pending")


class MockAudioProvider(AudioProvider):
    """Mock audio provider for testing."""
    
    async def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> Path:
        """Return mock audio path for testing."""
        mock_path = Path(f"outputs/media/mock_audio_{hash(text)}.wav")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()  # Create empty file for now
        return mock_path


class AudioFactory:
    """Factory for creating audio providers."""
    
    @staticmethod
    def create_provider(provider_type: str, api_key: Optional[str] = None) -> AudioProvider:
        """Create an audio provider instance."""
        if provider_type.lower() == "elevenlabs":
            if not api_key:
                raise ValueError("API key required for ElevenLabs")
            return ElevenLabsProvider(api_key)
        elif provider_type.lower() == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI TTS")
            return OpenAITTSProvider(api_key)
        elif provider_type.lower() == "mock":
            return MockAudioProvider()
        else:
            raise ValueError(f"Unsupported audio provider: {provider_type}")