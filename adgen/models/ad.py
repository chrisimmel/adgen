from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class AdConcept(BaseModel):
    """Represents the high-level concept for an advertisement."""
    
    target_audience: str = Field(description="Primary target demographic for the ad")
    key_message: str = Field(description="Core message to communicate")
    tone: str = Field(description="Tone of the advertisement (e.g., professional, playful, urgent)")
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
    
    scenes: List[str] = Field(description="List of visual scenes/shots")
    visual_style: str = Field(description="Overall visual style description")
    color_palette: List[str] = Field(description="Suggested color palette")
    text_overlays: List[str] = Field(description="Text overlays to display")


class AdAssets(BaseModel):
    """Represents generated media assets for the advertisement."""
    
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    music_path: Optional[Path] = None
    final_video_path: Optional[Path] = None


class AdProject(BaseModel):
    """Complete advertisement project containing all components."""
    
    # Input
    business_description: str
    
    # Generated content
    concept: Optional[AdConcept] = None
    script: Optional[AdScript] = None
    visual_plan: Optional[VisualPlan] = None
    assets: Optional[AdAssets] = None
    
    # Metadata
    project_id: str
    created_at: str
    status: str = "created"  # created, concept_generated, script_generated, etc.