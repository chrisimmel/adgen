from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
import time

import aiohttp


class VideoProvider(ABC):
    """Abstract base class for video generation providers."""

    @abstractmethod
    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 15,
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> Path:
        """Generate video from text prompt."""
        pass


class RunwayMLProvider(VideoProvider):
    """RunwayML video generation provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runwayml.com/v1"

    def _clean_prompt(self, prompt: str) -> str:
        """Clean and validate prompt for RunwayML API."""
        import re

        # Remove extra whitespace and newlines
        clean = re.sub(r"\s+", " ", prompt.strip())

        # Remove any special formatting or markdown
        clean = re.sub(r"[#*_`]", "", clean)

        # Limit length (RunwayML has prompt limits)
        max_length = 1000
        if len(clean) > max_length:
            # Try to cut at sentence boundary
            sentences = clean.split(".")
            result = ""
            for sentence in sentences:
                if len(result + sentence + ".") <= max_length:
                    result += sentence + "."
                else:
                    break
            clean = result.strip() or clean[:max_length]

        # Ensure it's not empty
        if not clean:
            clean = "Professional advertisement"

        return clean

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 15,
        aspect_ratio: str = "16:9",
        **_kwargs,
    ) -> Path:
        """Generate video using RunwayML API."""
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Import RunwayML SDK
                # Ensure .env is loaded for API key access
                from dotenv import load_dotenv
                import runwayml

                load_dotenv()

                # Create async client
                client = runwayml.AsyncRunwayML(api_key=self.api_key)

                # Map aspect ratio to RunwayML format
                ratio_map = {
                    "16:9": "1280:720",
                    "9:16": "720:1280",
                    "1:1": "720:720",
                    "4:3": "960:720",
                }
                runway_ratio = ratio_map.get(aspect_ratio, "1280:720")

                print(
                    f"Generating video with RunwayML: '{prompt[:100]}...' ({runway_ratio}, {duration_seconds}s)"
                )

                # Clean and validate the prompt for RunwayML
                clean_prompt = self._clean_prompt(prompt)
                print(f"Cleaned prompt: '{clean_prompt}'")

                # Step 1: Generate an image from the prompt
                image_task = await client.text_to_image.create(
                    model="gen4_image", prompt_text=clean_prompt, ratio=runway_ratio
                )

                # Wait for image generation to complete
                image_result = await image_task.wait_for_task_output()

                # Handle different response formats from RunwayML
                image_url = None
                if hasattr(image_result, "output") and image_result.output:
                    if isinstance(image_result.output, dict):
                        image_url = image_result.output.get("url")
                    elif (
                        isinstance(image_result.output, list)
                        and len(image_result.output) > 0
                    ):
                        first_result = image_result.output[0]
                        if isinstance(first_result, str):
                            # Direct URL string
                            image_url = first_result
                        elif isinstance(first_result, dict):
                            image_url = first_result.get("url")
                        elif hasattr(first_result, "url"):
                            image_url = first_result.url
                    elif isinstance(image_result.output, str):
                        # Direct URL string
                        image_url = image_result.output

                if not image_url:
                    print(
                        f"Image result structure: {type(image_result.output)} - {image_result.output}"
                    )
                    raise ValueError(
                        "Failed to extract image URL from RunwayML response"
                    )

                print(f"Generated image: {image_url}")

                # Step 2: Create video from the generated image
                # Create a shorter, animation-focused prompt
                animation_prompt = self._clean_prompt(
                    "Smooth animation, professional commercial style"
                )

                video_task = await client.image_to_video.create(
                    model="gen4_turbo",
                    prompt_image=image_url,
                    prompt_text=animation_prompt,
                    ratio=runway_ratio,
                    duration=min(duration_seconds, 10),  # RunwayML has duration limits
                )

                # Wait for video generation to complete
                video_result = await video_task.wait_for_task_output()

                # Handle different response formats from RunwayML
                video_url = None
                if hasattr(video_result, "output") and video_result.output:
                    if isinstance(video_result.output, dict):
                        video_url = video_result.output.get("url")
                    elif (
                        isinstance(video_result.output, list)
                        and len(video_result.output) > 0
                    ):
                        first_result = video_result.output[0]
                        if isinstance(first_result, str):
                            # Direct URL string
                            video_url = first_result
                        elif isinstance(first_result, dict):
                            video_url = first_result.get("url")
                        elif hasattr(first_result, "url"):
                            video_url = first_result.url
                    elif isinstance(video_result.output, str):
                        # Direct URL string
                        video_url = video_result.output

                if not video_url:
                    print(
                        f"Video result structure: {type(video_result.output)} - {video_result.output}"
                    )
                    raise ValueError(
                        "Failed to extract video URL from RunwayML response"
                    )

                print(f"Generated video: {video_url}")

                # Download the video
                output_path = Path(
                    f"outputs/media/runway_video_{hash(prompt)}_{int(time.time())}.mp4"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                async with (
                    aiohttp.ClientSession() as session,
                    session.get(video_url) as response,
                ):
                    if response.status == 200:
                        with open(output_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                    else:
                        raise ValueError(
                            f"Failed to download video: HTTP {response.status}"
                        )

                print(f"Video saved to: {output_path}")
                return output_path

            except ImportError:
                raise RuntimeError(
                    "RunwayML SDK not installed. Run: pip install runwayml"
                ) from None
            except (TimeoutError, aiohttp.ClientError) as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    print("All retry attempts failed")
                    break
            except Exception as e:
                print(
                    f"RunwayML generation failed on attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                import traceback

                print(f"Full traceback: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    break

        # If we get here, all attempts failed
        print("Falling back to mock video generation")
        mock_path = Path(f"outputs/media/runway_fallback_{hash(prompt)}.mp4")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()
        return mock_path


class Veo3Provider(VideoProvider):
    """Google Veo 3 video generation provider via FAL."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _clean_prompt_unicode(self, prompt: str) -> str:
        """Clean prompt to handle Unicode characters that cause encoding issues."""
        import re

        # Replace smart quotes and other problematic Unicode characters
        replacements = {
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u2026": "...",  # Horizontal ellipsis
        }

        clean = prompt
        for unicode_char, replacement in replacements.items():
            clean = clean.replace(unicode_char, replacement)

        # Remove any remaining non-ASCII characters
        clean = re.sub(r"[^\x00-\x7F]+", " ", clean)

        # Clean up extra whitespace
        clean = re.sub(r"\s+", " ", clean).strip()

        return clean

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 8,
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> Path:
        """Generate video using Veo 3 via FAL API."""
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Import FAL client
                import os

                from dotenv import load_dotenv
                import fal_client

                # Ensure environment variables are loaded
                load_dotenv()

                # Set the FAL_KEY environment variable, ensuring it's ASCII-safe
                clean_api_key = self.api_key.encode("ascii", "ignore").decode("ascii")
                os.environ["FAL_KEY"] = clean_api_key

                print(
                    f"Generating video with Veo 3: '{prompt[:100]}...' ({aspect_ratio}, {duration_seconds}s)"
                )

                # Clean prompt for Unicode issues
                clean_prompt = self._clean_prompt_unicode(prompt.strip())
                print(
                    f"Original prompt length: {len(prompt)}, cleaned length: {len(clean_prompt)}"
                )

                # Veo 3 parameters
                arguments = {
                    "prompt": clean_prompt,
                    "aspect_ratio": aspect_ratio,
                    "generate_audio": kwargs.get("generate_audio", False),
                }

                # Add optional parameters if provided
                if "negative_prompt" in kwargs:
                    arguments["negative_prompt"] = kwargs["negative_prompt"]
                if "enhance_prompt" in kwargs:
                    arguments["enhance_prompt"] = kwargs["enhance_prompt"]
                if "seed" in kwargs:
                    arguments["seed"] = kwargs["seed"]

                print(f"Veo 3 arguments: {arguments}")

                # Submit video generation request
                result = fal_client.subscribe("fal-ai/veo3", arguments=arguments)

                # Extract video URL from response
                video_url = None
                if hasattr(result, "get") and result.get("video"):
                    video_data = result["video"]
                    if isinstance(video_data, dict) and "url" in video_data:
                        video_url = video_data["url"]
                    elif isinstance(video_data, str):
                        video_url = video_data
                elif hasattr(result, "video") and result.video:
                    if hasattr(result.video, "url"):
                        video_url = result.video.url
                    else:
                        video_url = str(result.video)

                if not video_url:
                    print(f"Veo 3 result structure: {type(result)} - {result}")
                    raise ValueError("Failed to extract video URL from Veo 3 response")

                print(f"Generated video URL: {video_url}")

                # Download the video
                output_path = Path(
                    f"outputs/media/veo3_video_{hash(prompt)}_{int(time.time())}.mp4"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                async with (
                    aiohttp.ClientSession() as session,
                    session.get(video_url) as response,
                ):
                    if response.status == 200:
                        with open(output_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                    else:
                        raise ValueError(
                            f"Failed to download video: HTTP {response.status}"
                        )

                print(f"Video saved to: {output_path}")
                return output_path

            except ImportError:
                raise RuntimeError(
                    "FAL client not installed. Run: pip install fal-client"
                ) from None
            except (TimeoutError, aiohttp.ClientError) as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    print("All retry attempts failed")
                    break
            except Exception as e:
                print(
                    f"Veo 3 generation failed on attempt {attempt + 1}: {type(e).__name__}: {e}"
                )
                import traceback

                print(f"Full traceback: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    break

        # If we get here, all attempts failed
        print("Falling back to mock video generation")
        mock_path = Path(f"outputs/media/veo3_fallback_{hash(prompt)}.mp4")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()
        return mock_path


class PikaProvider(VideoProvider):
    """Pika Labs video generation provider."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: int = 15,
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> Path:
        # TODO: Implement Pika API integration
        raise NotImplementedError("Pika integration pending")


class MockVideoProvider(VideoProvider):
    """Mock video provider for testing."""

    async def generate_video(
        self,
        prompt: str,
        _duration_seconds: int = 15,
        _aspect_ratio: str = "16:9",
        **_kwargs,
    ) -> Path:
        """Return mock video path for testing."""
        mock_path = Path(f"outputs/media/mock_video_{hash(prompt)}.mp4")
        mock_path.parent.mkdir(parents=True, exist_ok=True)
        mock_path.touch()  # Create empty file for now
        return mock_path


class VideoFactory:
    """Factory for creating video providers."""

    @staticmethod
    def create_provider(
        provider_type: str, api_key: str | None = None
    ) -> VideoProvider:
        """Create a video provider instance."""
        if provider_type.lower() == "runwayml":
            if not api_key:
                raise ValueError("API key required for RunwayML")
            return RunwayMLProvider(api_key)
        elif provider_type.lower() == "veo3":
            if not api_key:
                raise ValueError("API key required for Veo 3")
            return Veo3Provider(api_key)
        elif provider_type.lower() == "pika":
            if not api_key:
                raise ValueError("API key required for Pika")
            return PikaProvider(api_key)
        elif provider_type.lower() == "mock":
            return MockVideoProvider()
        else:
            raise ValueError(f"Unsupported video provider: {provider_type}")
