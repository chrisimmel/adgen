from pathlib import Path

from pydantic import BaseModel, Field


class AdConcept(BaseModel):
    """Represents the high-level concept for an advertisement."""

    target_audience: str = Field(description="Primary target demographic for the ad")
    key_message: str = Field(description="Core message to communicate")
    tone: str = Field(
        description="Tone of the advertisement (e.g., professional, playful, urgent)"
    )
    style: str = Field(description="Visual style (e.g., modern, minimalist, energetic)")
    call_to_action: str = Field(description="What action should viewers take")
    emotional_appeal: str = Field(description="Primary emotion to evoke")


class AdScript(BaseModel):
    """Represents the script/narration for an advertisement."""

    hook: str = Field(description="Opening line to grab attention")
    main_content: str = Field(description="Main body of the script")
    call_to_action: str = Field(description="Closing call to action")
    total_words: int = Field(description="Total word count")
    estimated_duration: float = Field(description="Estimated duration in seconds")


class VisualPlan(BaseModel):
    """Represents the visual plan for the advertisement."""

    scenes: list[str] = Field(description="List of visual scenes/shots")
    visual_style: str = Field(description="Overall visual style description")
    color_palette: list[str] = Field(description="Suggested color palette")
    text_overlays: list[str] = Field(description="Text overlays to display")


class AdAssets(BaseModel):
    """Represents generated media assets for the advertisement."""

    video_path: Path | None = None  # Legacy single video path
    scene_clips: list[Path] | None = None  # Individual scene video clips
    audio_path: Path | None = None
    music_path: Path | None = None
    final_video_path: Path | None = None  # Composed final video from all scenes


class AdProject(BaseModel):
    """Complete advertisement project containing all components."""

    # Input - either URL or direct business description
    source_url: str | None = None
    business_description: str | None = None

    # Intermediate content (for URL input)
    web_content_markdown: str | None = None

    # Generated content
    concept: AdConcept | None = None
    script: AdScript | None = None
    visual_plan: VisualPlan | None = None
    assets: AdAssets | None = None

    # Metadata
    project_id: str
    created_at: str
    status: str = (
        "created"  # created, web_scraped, business_analyzed, concept_generated, etc.
    )
