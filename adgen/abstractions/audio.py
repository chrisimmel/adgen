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
        self, text: str, voice_id: str | None = "Rachel", speed: float = 1.0, **kwargs
    ) -> Path:
        """Generate speech using ElevenLabs TTS API."""
        max_retries = 3
        retry_delay = 2

        try:
            import asyncio

            from elevenlabs import ElevenLabs
            from elevenlabs.types import VoiceSettings

            # Initialize ElevenLabs client
            client = ElevenLabs(api_key=self.api_key)
        except Exception as e:
            print(f"Error initializing ElevenLabs client: {e}")
            raise e

        for attempt in range(max_retries):
            try:
                # Map voice names to IDs if needed
                voice_mapping = {
                    "rachel": "21m00Tcm4TlvDq8ikWAM",
                    "drew": "29vD33N1CtxCmqQRPOHJ",
                    "clyde": "2EiwWnXFnvU5JabPnv8n",
                    "paul": "5Q0t7uMcjvnagumLfvZi",
                    "domi": "AZnzlk1XvdvUeBnXmlld",
                    "dave": "CYw3kZ02Hs0563khs1Fj",
                    "fin": "D38z5RcWu1voky8WS1ja",
                    "sarah": "EXAVITQu4vr4xnSDxMaL",
                    "antoni": "ErXwobaYiN019PkySvjV",
                    "thomas": "GBv7mTt0atIp3Br8iCZE",
                    "charlotte": "IKne3meq5aSn9XLyUdCD",
                    "matilda": "XrExE9yKIg1WjnnlVkGX",
                    "matthew": "Yko7PKHZNXotIFUBG7I9",
                    "james": "ZQe5CqHNLWdM6q4Ej6O7",
                    "joseph": "Zlb1dXrM653N07WRdFW3",
                    "jeremy": "bVMeCyTHy58xNoL34h3p",
                    "michael": "flq6f7yk4E4fJM5XTYuZ",
                    "ethan": "g5CIjZEefAph4nQFvHAz",
                    "gigi": "jBpfuIE2acCO8z3wKNLl",
                    "freya": "jsCqWAovK2LkecY7zXl4",
                    "grace": "oWAxZDx7w5VEj9dCyTzz",
                    "daniel": "onwK4e9ZLuTAKqWW03F9",
                    "lily": "pFGS58b6Hhp6WmNhAQwR",
                    "serena": "pMsXgVXv3BLzUgSXRplE",
                    "adam": "pNInz6obpgDQGcFmaJgB",
                    "nicole": "piTKgcLEGmPE4e6mEKli",
                    "jessie": "t0jbNlBVZ17f02VDIeMI",
                    "ryan": "wViXBPUzp2ZZixB1xQuM",
                    "sam": "yoZ06aMxZJJ28mfd3POQ",
                    "glinda": "z9fAnlkpzviPz146aGWa",
                }

                # Get voice ID (support both name and direct ID)
                final_voice_id = voice_id
                if voice_id and voice_id.lower() in voice_mapping:
                    final_voice_id = voice_mapping[voice_id.lower()]
                elif not voice_id:
                    final_voice_id = voice_mapping["rachel"]

                print(
                    f"Generating audio with ElevenLabs TTS: voice={voice_id}({final_voice_id[:8]}...), speed={speed}"
                )
                print(f"Text: '{text[:100]}...'")

                # Configure voice settings
                # ElevenLabs uses stability and similarity_boost instead of speed
                # Convert speed to stability (inverse relationship for naturalness)
                stability = max(0.0, min(1.0, 1.0 - (speed - 1.0) * 0.3))
                similarity_boost = kwargs.get("similarity_boost", 0.75)

                voice_settings = VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=kwargs.get("style", 0.0),
                    use_speaker_boost=kwargs.get("use_speaker_boost", True),
                )

                print(
                    f"Voice settings: stability={stability:.2f}, similarity_boost={similarity_boost}"
                )

                # Generate audio (this is a blocking call, so we'll run it in executor)
                loop = asyncio.get_event_loop()

                # Create a function with bound parameters to avoid lambda closure issues
                def generate_audio(
                    voice_id=final_voice_id,
                    settings=voice_settings,
                    model_id=kwargs.get("model_id", "eleven_multilingual_v2"),
                ):
                    return client.text_to_speech.convert(
                        voice_id=voice_id,
                        text=text,
                        voice_settings=settings,
                        model_id=model_id,
                    )

                audio_generator = await loop.run_in_executor(None, generate_audio)

                # Save audio to file
                output_path = Path(
                    f"outputs/media/elevenlabs_audio_{hash(text)}_{int(time.time())}.mp3"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Write audio data
                with open(output_path, "wb") as f:
                    for chunk in audio_generator:
                        f.write(chunk)

                print(f"Audio saved to: {output_path}")
                return output_path

            except ImportError:
                raise RuntimeError(
                    "ElevenLabs SDK not installed. Run: uv add elevenlabs"
                ) from None
            except Exception as e:
                print(
                    f"ElevenLabs TTS generation failed on attempt {attempt + 1}: {type(e).__name__}: {e}"
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
        mock_path = Path(f"outputs/media/elevenlabs_audio_fallback_{hash(text)}.mp3")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()
        return mock_path


class OpenAITTSProvider(AudioProvider):
    """OpenAI text-to-speech provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_speech(
        self, text: str, voice_id: str | None = "alloy", speed: float = 1.0, **_kwargs
    ) -> Path:
        """Generate speech using OpenAI TTS API."""

        try:
            from dotenv import load_dotenv
            from openai import AsyncOpenAI

            # Ensure environment variables are loaded
            load_dotenv()

            # Create async client
            client = AsyncOpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise e

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                voice = voice_id or "alloy"
                print(f"Generating audio with OpenAI TTS: voice={voice}, speed={speed}")
                print(f"Text: '{text[:100]}...'")

                # Generate audio with higher quality model
                response = await client.audio.speech.create(
                    model="tts-1-hd",  # Higher quality for better engagement
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
                    f.write(response.content)

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
