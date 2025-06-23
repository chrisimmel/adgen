from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
import time


class AudioProvider(ABC):
    """Abstract base class for text-to-speech providers."""

    @abstractmethod
    async def generate_speech(
        self, text: str, voice_id: str | None = None, speed: float = 1.0, **kwargs
    ) -> Path:
        """Generate speech audio from text."""
        pass


class ElevenLabsProvider(AudioProvider):
    """ElevenLabs text-to-speech provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_speech(
        self, text: str, voice_id: str | None = None, speed: float = 1.0, **kwargs
    ) -> Path:
        # TODO: Implement ElevenLabs API integration
        raise NotImplementedError("ElevenLabs integration pending")


class OpenAITTSProvider(AudioProvider):
    """OpenAI text-to-speech provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_speech(
        self, text: str, voice_id: str | None = "alloy", speed: float = 1.0, **_kwargs
    ) -> Path:
        """Generate speech using OpenAI TTS API."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                from dotenv import load_dotenv
                from openai import AsyncOpenAI

                # Ensure environment variables are loaded
                load_dotenv()

                # Create async client
                client = AsyncOpenAI(api_key=self.api_key)

                voice = voice_id or "alloy"
                print(f"Generating audio with OpenAI TTS: voice={voice}, speed={speed}")
                print(f"Text: '{text[:100]}...'")

                # Generate audio
                response = await client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    speed=speed,
                )

                # Save audio to file
                output_path = Path(
                    f"outputs/media/openai_audio_{hash(text)}_{int(time.time())}.mp3"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Write audio data
                with open(output_path, "wb") as f:
                    async for chunk in response.iter_bytes():
                        f.write(chunk)

                print(f"Audio saved to: {output_path}")
                return output_path

            except ImportError:
                raise RuntimeError(
                    "OpenAI SDK not installed. Run: pip install openai"
                ) from None
            except Exception as e:
                print(
                    f"OpenAI TTS generation failed on attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    break

        # If we get here, all attempts failed
        print("Falling back to mock audio generation")
        mock_path = Path(f"outputs/media/openai_audio_fallback_{hash(text)}.mp3")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()
        return mock_path


class MockAudioProvider(AudioProvider):
    """Mock audio provider for testing."""

    async def generate_speech(
        self, text: str, _voice_id: str | None = None, _speed: float = 1.0, **_kwargs
    ) -> Path:
        """Return mock audio path for testing."""
        mock_path = Path(f"outputs/media/mock_audio_{hash(text)}.wav")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()  # Create empty file for now
        return mock_path


class AudioFactory:
    """Factory for creating audio providers."""

    @staticmethod
    def create_provider(
        provider_type: str, api_key: str | None = None
    ) -> AudioProvider:
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
