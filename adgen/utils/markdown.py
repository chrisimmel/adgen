from datetime import datetime
from pathlib import Path

from adgen.models.ad import AdConcept, AdProject, AdScript, VisualPlan


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

## Business Description
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
