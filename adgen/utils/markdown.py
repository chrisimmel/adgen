from datetime import datetime
from pathlib import Path

import html2text
import httpx

from adgen.models.ad import AdConcept, AdProject, AdScript, VisualPlan


async def get_markdown_from_web_page(
    url: str, ignore_links: bool = True, ignore_images: bool = True, timeout: int = 20
) -> str | None:
    """Convert HTML from a URL to nicely formatted markdown (async version).

    Args:
        url: The URL to fetch and convert
        ignore_links: If True, don't include links in the markdown output
        ignore_images: If True, don't include images in the markdown output
        timeout: Request timeout in seconds

    Returns:
        Markdown string or None if conversion failed
    """
    # Fetch the HTML with proper headers using httpx
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        # Configure the converter
        h = html2text.HTML2Text()
        h.ignore_links = ignore_links
        h.ignore_images = ignore_images
        h.body_width = 0  # Don't wrap lines
        h.unicode_snob = True
        h.bypass_tables = False
        h.ignore_emphasis = False  # Keep bold/italic formatting
        h.mark_code = True  # Mark code blocks

        # Convert and clean up
        markdown = h.handle(response.text)
        return markdown.strip()


def generate_concept_markdown(concept: AdConcept, project_id: str) -> str:
    """Generate markdown for the ad concept."""

    return f"""# Ad Concept: {project_id}

## Target Audience
{concept.target_audience}

## Key Message
{concept.key_message}

## Tone & Style
- **Tone**: {concept.tone}
- **Visual Style**: {concept.style}

## Emotional Appeal
{concept.emotional_appeal}

## Call to Action
{concept.call_to_action}

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""


def generate_script_markdown(script: AdScript, project_id: str) -> str:
    """Generate markdown for the ad script."""

    return f"""# Ad Script: {project_id}

## Hook (Opening)
> {script.hook}

## Main Content
{script.main_content}

## Call to Action (Closing)
> {script.call_to_action}

## Script Details
- **Total Words**: {script.total_words}
- **Estimated Duration**: {script.estimated_duration} seconds

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""


def generate_visual_plan_markdown(visual_plan: VisualPlan, project_id: str) -> str:
    """Generate markdown for the visual plan."""

    scenes_list = "\n".join(
        [f"{i+1}. {scene}" for i, scene in enumerate(visual_plan.scenes)]
    )
    colors_list = "\n".join([f"- {color}" for color in visual_plan.color_palette])
    overlays_list = "\n".join([f"- {overlay}" for overlay in visual_plan.text_overlays])

    return f"""# Visual Plan: {project_id}

## Overall Visual Style
{visual_plan.visual_style}

## Scenes
{scenes_list}

## Color Palette
{colors_list}

## Text Overlays
{overlays_list}

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""


def generate_full_project_markdown(project: AdProject) -> str:
    """Generate comprehensive markdown for the entire project."""

    content = f"""# Ad Project: {project.project_id}

**Created**: {project.created_at}
**Status**: {project.status}

"""

    if project.source_url:
        content += f"""## Source URL
{project.source_url}

"""

    if project.business_description:
        content += f"""## Business Description
{project.business_description}

"""

    if project.concept:
        content += f"""## Concept
- **Target Audience**: {project.concept.target_audience}
- **Key Message**: {project.concept.key_message}
- **Tone**: {project.concept.tone}
- **Visual Style**: {project.concept.style}
- **Emotional Appeal**: {project.concept.emotional_appeal}
- **Call to Action**: {project.concept.call_to_action}

"""

    if project.script:
        content += f"""## Script
### Hook
> {project.script.hook}

### Main Content
{project.script.main_content}

### Call to Action
> {project.script.call_to_action}

**Details**: {project.script.total_words} words, ~{project.script.estimated_duration}s

"""

    if project.visual_plan:
        scenes_list = "\n".join(
            [f"{i+1}. {scene}" for i, scene in enumerate(project.visual_plan.scenes)]
        )
        content += f"""## Visual Plan
### Style
{project.visual_plan.visual_style}

### Scenes
{scenes_list}

### Color Palette
{', '.join(project.visual_plan.color_palette)}

### Text Overlays
{', '.join(project.visual_plan.text_overlays)}

"""

    if project.assets:
        content += """## Generated Assets
"""
        if project.assets.video_path:
            content += f"- **Video**: {project.assets.video_path}\n"
        if project.assets.audio_path:
            content += f"- **Audio**: {project.assets.audio_path}\n"
        if project.assets.music_path:
            content += f"- **Music**: {project.assets.music_path}\n"
        if project.assets.final_video_path:
            content += f"- **Final Video**: {project.assets.final_video_path}\n"

    content += f"""
---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    return content


def save_markdown(content: str, filename: str, output_dir: str = "outputs") -> Path:
    """Save markdown content to file."""
    output_path = Path(output_dir) / "concepts" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path
