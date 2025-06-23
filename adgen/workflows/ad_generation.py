from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from adgen.abstractions.video import VideoFactory
from adgen.models.ad import AdAssets, AdConcept, AdProject, AdScript, VisualPlan
from adgen.utils.config import Config, get_api_key
from adgen.utils.markdown import get_markdown_from_web_page


class AdGenerationState(dict[str, Any]):
    """State object for the ad generation workflow."""

    project: AdProject
    config: Config
    approve_concept: bool = False
    approve_final: bool = False


def create_llm(config: Config) -> BaseChatModel:
    """Create LLM instance based on configuration."""
    provider = config.providers["llm"]
    api_key = get_api_key(provider, "llm")

    if not api_key:
        raise ValueError(f"API key not found for {provider}")

    if provider == "openai":
        model = config.llm.get("openai", {}).get("model", "gpt-4")
        temperature = config.llm.get("openai", {}).get("temperature", 0.7)
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    elif provider == "anthropic":
        model = config.llm.get("anthropic", {}).get("model", "claude-3-sonnet-20240229")
        temperature = config.llm.get("anthropic", {}).get("temperature", 0.7)
        return ChatAnthropic(api_key=api_key, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def scrape_web_content_node(state: AdGenerationState) -> AdGenerationState:
    """Extract markdown content from the provided URL."""
    if not state["project"].source_url:
        # Skip if no URL provided
        return state

    print(f"Scraping content from: {state['project'].source_url}")

    try:
        markdown_content = await get_markdown_from_web_page(state["project"].source_url)
        state["project"].web_content_markdown = markdown_content
        state["project"].status = "web_scraped"
        print(f"Successfully scraped {len(markdown_content)} characters of content")
    except Exception as e:
        print(f"Error scraping web content: {e}")
        # Continue with empty content for now
        state["project"].web_content_markdown = ""
        state["project"].status = "web_scrape_failed"

    return state


async def generate_business_description_node(
    state: AdGenerationState,
) -> AdGenerationState:
    """Generate a business description from scraped web content."""
    if not state["project"].web_content_markdown:
        # Skip if no web content
        return state

    llm = create_llm(state["config"])

    prompt = f"""
    You are a business analyst. Analyze the following website content and create a comprehensive business description that will be used for creating a video advertisement.

    Website Content (Markdown):
    {state["project"].web_content_markdown[:8000]}  # Limit to avoid token limits

    Please create a detailed business description that includes:

    ## Company Overview
    - What kind of business this is
    - Their mission and values
    - Their unique value proposition

    ## Branding & Style
    - Brand personality and tone
    - Visual style preferences (if evident)
    - Brand positioning

    ## Products & Services
    - Main product categories
    - Key products or services
    - Target market for each

    ## Customer Focus
    - Primary customer demographics
    - Customer pain points they solve
    - Customer success stories or testimonials (if mentioned)

    Format this as a well-structured markdown document that captures the essence of the business for advertising purposes. Focus on details that would be relevant for creating a compelling video advertisement.
    """

    print("Generating business description from web content...")

    try:
        description = await llm.ainvoke(prompt)
        state["project"].business_description = description.content
        state["project"].status = "business_analyzed"
        print(f"Generated business description: {len(description.content)} characters")
    except Exception as e:
        print(f"Error generating business description: {e}")
        # Fallback to truncated web content
        state["project"].business_description = state["project"].web_content_markdown[
            :2000
        ]
        state["project"].status = "business_analysis_failed"

    return state


def should_scrape_web_content(state: AdGenerationState) -> str:
    """Decide whether to scrape web content or go directly to concept generation."""
    if state["project"].source_url:
        return "scrape_web"
    elif state["project"].business_description:
        return "generate_concept"
    else:
        raise ValueError("Either source_url or business_description must be provided")


def should_generate_business_description(state: AdGenerationState) -> str:
    """Decide whether to generate business description from web content."""
    if (
        state["project"].web_content_markdown
        and state["project"].status == "web_scraped"
    ):
        return "generate_business_description"
    else:
        return "generate_concept"


async def generate_concept_node(state: AdGenerationState) -> AdGenerationState:
    """Generate the ad concept from business description."""
    llm = create_llm(state["config"]).with_structured_output(AdConcept)

    prompt = f"""
    You are an expert advertising strategist. Create a comprehensive ad concept for the following business:

    Business Description: {state["project"].business_description}

    Target Duration: {state["config"].ad_duration_seconds} seconds

    Please provide a detailed ad concept including:
    - Target audience
    - Key message
    - Tone
    - Visual style
    - Call to action
    - Emotional appeal

    Remember that we are trying to sell the products or services of the business. The target
    audience for the ad depends on the who the business is and what they sell, but is whoever
    the business sells to.

    Make it compelling and appropriate for a {state["config"].ad_duration_seconds}-second video ad.
    """

    print(f"Generating concept with prompt: {prompt}")

    concept = await llm.ainvoke(prompt)

    state["project"].concept = concept
    state["project"].status = "concept_generated"

    print(f"Concept generated: {concept}")
    return state


async def review_concept_node(state: AdGenerationState) -> AdGenerationState:
    """Human review of the generated concept."""
    # This is a placeholder - actual approval is handled by CLI
    # The approve_concept flag will be set by the CLI after human review
    # Don't auto-approve here
    return state


async def generate_script_node(state: AdGenerationState) -> AdGenerationState:
    """Generate the ad script based on the approved concept."""
    if not state["approve_concept"]:
        return state

    llm = create_llm(state["config"]).with_structured_output(AdScript)
    concept = state["project"].concept

    prompt = f"""
    Create a compelling {state["config"].ad_duration_seconds}-second video ad script based on this concept:

    Target Audience: {concept.target_audience}
    Key Message: {concept.key_message}
    Tone: {concept.tone}
    Call to Action: {concept.call_to_action}

    The script should:
    - Have a strong hook in the first 3 seconds
    - Deliver the key message clearly
    - End with a clear call to action
    - Be exactly the right length for {state["config"].ad_duration_seconds} seconds (approximately {state["config"].ad_duration_seconds * 2.5} words)

    Format the response with:
    Hook: [opening line]
    Main Content: [body of script]
    Call to Action: [closing]
    """

    print(f"Generating script with prompt: {prompt}")

    script = await llm.ainvoke(prompt)

    state["project"].script = script
    state["project"].status = "script_generated"

    print(f"Script generated: {script}")
    return state


async def generate_visual_plan_node(state: AdGenerationState) -> AdGenerationState:
    """Generate visual plan for the advertisement."""
    if not state["approve_concept"]:
        return state

    llm = create_llm(state["config"]).with_structured_output(VisualPlan)
    concept = state["project"].concept
    script = state["project"].script

    config = state["config"]
    provider_type = config.providers.get("video", "mock")
    provider_config = config.video.get(provider_type, {})
    max_scenes = provider_config.get("max_scenes", 5)

    prompt = f"""
    Create a detailed visual plan for a {state["config"].ad_duration_seconds}-second video ad:

    Concept Style: {concept.style}
    Script: {script.hook} {script.main_content} {script.call_to_action}

    Provide:
    - 3-{max_scenes} specific visual scenes/shots
    - Overall visual style description
    - Color palette (3-5 colors)
    - Text overlays to display
    """

    print(f"Generating visual plan with prompt: {prompt}")

    visual_plan = await llm.ainvoke(prompt)

    state["project"].visual_plan = visual_plan
    state["project"].status = "visual_plan_generated"

    print(f"Visual plan generated: {visual_plan}")
    return state


def _create_comprehensive_prompt(concept, script, visual_plan, duration_seconds):
    """Create a comprehensive prompt for single-clip video generation."""
    scenes_text = (
        ", then ".join(visual_plan.scenes) if visual_plan.scenes else "product showcase"
    )

    comprehensive_prompt = (
        f"{concept.style} {concept.tone} {duration_seconds}-second commercial advertisement. "
        f"Opening: {script.hook}. "
        f"Main content: {script.main_content}. "
        f"Visual sequence: {scenes_text}. "
        f"Ending: {script.call_to_action}. "
        f"Key message: {concept.key_message}. "
        f"Professional commercial cinematography with smooth transitions between scenes."
    )

    return comprehensive_prompt


async def generate_video_node(state: AdGenerationState) -> AdGenerationState:
    """Generate multiple scene-based video clips using the visual plan and script."""
    if not state["approve_concept"]:
        return state

    # Ensure environment variables are loaded
    from dotenv import load_dotenv

    load_dotenv()

    concept = state["project"].concept
    script = state["project"].script
    visual_plan = state["project"].visual_plan

    if not concept or not script or not visual_plan:
        print("Missing required components for video generation")
        return state

    # Create video provider
    config = state["config"]
    provider_type = config.providers.get("video", "mock")
    api_key = get_api_key(provider_type, "video")

    print(f"Video provider: {provider_type}")
    print(f"API key available: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API key length: {len(api_key)}")

    scene_clips = []

    try:
        video_provider = VideoFactory.create_provider(provider_type, api_key)

        # Get provider-specific constraints
        provider_config = config.video.get(provider_type, {})
        single_clip_mode = provider_config.get("single_clip_mode", False)

        if single_clip_mode:
            # Generate entire sequence as a single comprehensive clip
            print(
                f"Generating single comprehensive video clip (provider: {provider_type})"
            )

            target_duration = config.ad_duration_seconds

            # Create comprehensive prompt that includes all scenes
            comprehensive_prompt = _create_comprehensive_prompt(
                concept, script, visual_plan, target_duration
            )

            print(f"Comprehensive prompt: {comprehensive_prompt[:200]}...")

            # Generate single video with provider-specific options
            video_kwargs = {
                "duration_seconds": target_duration,
                "aspect_ratio": config.video.get("aspect_ratio", "16:9"),
            }

            # Add provider-specific options
            if provider_type == "veo3":
                video_kwargs["generate_audio"] = provider_config.get(
                    "generate_audio", False
                )

            scene_path = await video_provider.generate_video(
                prompt=comprehensive_prompt.strip(), **video_kwargs
            )

            scene_clips.append(scene_path)
            print(f"Single comprehensive video generated: {scene_path}")

        else:
            # Multi-scene mode (existing logic)
            min_duration = provider_config.get("min_duration", 3)
            max_duration = provider_config.get("max_duration", 10)

            # Limit scenes based on provider constraints and target duration
            scenes = visual_plan.scenes if visual_plan.scenes else ["product showcase"]

            # Calculate optimal scene count to fit within target duration
            target_duration = config.ad_duration_seconds

            # Disable limiting scenes for now. We will instead trim them in the compose
            # step if we need to.

            # optimal_scenes = min(
            #     max_scenes, target_duration // min_duration, len(scenes)
            # )

            # # Select the most important scenes if we need to limit
            # if len(scenes) > optimal_scenes:
            #     print(
            #         f"Limiting to {optimal_scenes} scenes (from {len(scenes)}) to fit {target_duration}s target"
            #     )
            #     scenes = scenes[:optimal_scenes]

            total_scenes = len(scenes)
            scene_duration = min(
                max_duration, max(min_duration, target_duration // total_scenes)
            )

            print(
                f"Generating {total_scenes} scene clips, {scene_duration}s each (provider: {provider_type})"
            )

            for i, scene in enumerate(scenes):
                print(f"Generating scene {i+1}/{total_scenes}: {scene[:50]}...")

                # Create scene-specific prompt
                scene_prompt = f"{concept.style} {concept.tone} commercial scene: {scene}. {concept.key_message}."

                # Generate scene clip with provider-specific options
                video_kwargs = {
                    "duration_seconds": scene_duration,
                    "aspect_ratio": config.video.get("aspect_ratio", "16:9"),
                }

                # Add provider-specific options
                if provider_type == "veo3":
                    video_kwargs["generate_audio"] = provider_config.get(
                        "generate_audio", False
                    )

                scene_path = await video_provider.generate_video(
                    prompt=scene_prompt.strip(), **video_kwargs
                )

                scene_clips.append(scene_path)
                print(f"Scene {i+1} generated: {scene_path}")

        # Update project with generated scene clips
        if not state["project"].assets:
            state["project"].assets = AdAssets()

        state["project"].assets.scene_clips = scene_clips
        state["project"].status = "scene_clips_generated"

        print(f"All {len(scene_clips)} scene clips generated successfully")

    except Exception as e:
        print(f"Scene video generation failed: {e}")
        # Continue without video for now
        state["project"].status = "video_generation_failed"

    return state


async def compose_video_node(state: AdGenerationState) -> AdGenerationState:
    """Compose individual scene clips into a final video using MoviePy."""
    if not state["approve_concept"]:
        return state

    # Check if we have scene clips to compose
    if not state["project"].assets or not state["project"].assets.scene_clips:
        print("No scene clips found to compose")
        return state

    scene_clips = state["project"].assets.scene_clips
    if len(scene_clips) <= 1:
        print("Single clip mode - using clip as final video")
        # If we only have one clip (single-clip mode), use it as the final video
        if scene_clips:
            state["project"].assets.final_video_path = scene_clips[0]
            state["project"].status = "video_composed"
        return state

    print(f"Composing {len(scene_clips)} scene clips into final video...")

    try:
        from pathlib import Path
        import time

        from moviepy import concatenate_videoclips
        from moviepy.video.io.VideoFileClip import VideoFileClip

        # Load all scene clips
        clips = []
        for clip_path in scene_clips:
            if clip_path.exists() and clip_path.stat().st_size > 0:
                clip = VideoFileClip(str(clip_path))
                clips.append(clip)
                print(f"Loaded clip: {clip_path} (duration: {clip.duration}s)")
            else:
                print(f"Warning: Skipping invalid clip: {clip_path}")

        if not clips:
            print("No valid clips to compose")
            state["project"].status = "video_composition_failed"
            return state

        # Calculate total duration and trim if necessary
        target_duration = state["config"].ad_duration_seconds
        total_duration = sum(clip.duration for clip in clips)

        print(f"Total clip duration: {total_duration}s, target: {target_duration}s")

        if total_duration > target_duration:
            print(f"Trimming clips to fit {target_duration}s target duration")

            # Calculate proportional duration for each clip
            scale_factor = target_duration / total_duration
            trimmed_clips = []

            for i, clip in enumerate(clips):
                # Calculate new duration for this clip
                new_duration = clip.duration * scale_factor

                # Preserve minimum durations for key scenes
                if i == 0 or i == len(clips) - 1:
                    # First and last scenes get at least 2 seconds
                    new_duration = max(2.0, new_duration)
                else:
                    # Middle scenes get at least 1 second
                    new_duration = max(1.0, new_duration)

                # Don't extend clips beyond their original duration
                new_duration = min(new_duration, clip.duration)

                # Trim the clip to the new duration
                trimmed_clip = clip.subclipped(0, new_duration)
                trimmed_clips.append(trimmed_clip)

                print(
                    f"Clip {i+1}: {clip.duration:.1f}s -> {trimmed_clip.duration:.1f}s"
                )

            clips = trimmed_clips
            actual_duration = sum(clip.duration for clip in clips)
            print(f"Final trimmed duration: {actual_duration:.1f}s")

        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)

        # Create output path for final video
        project_id = state["project"].project_id
        timestamp = int(time.time())
        final_video_path = Path(f"outputs/media/{project_id}_final_{timestamp}.mp4")
        final_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the final video
        print(f"Writing final video to: {final_video_path}")
        final_clip.write_videofile(
            str(final_video_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            # verbose=False,
            logger=None,  # Suppress MoviePy logs
        )

        # Clean up clips from memory
        for clip in clips:
            clip.close()
        final_clip.close()

        # Update project with final video path
        state["project"].assets.final_video_path = final_video_path
        state["project"].status = "video_composed"

        print(f"Final video composed successfully: {final_video_path}")
        # Check final video duration
        with VideoFileClip(str(final_video_path)) as final_check:
            print(f"Final video duration: {final_check.duration}s")

    except ImportError:
        print("MoviePy not available. Install with: pip install moviepy")
        state["project"].status = "video_composition_failed"
    except Exception as e:
        print(f"Video composition failed: {e}")
        import traceback

        print(f"Full traceback: {traceback.format_exc()}")
        state["project"].status = "video_composition_failed"

    return state


def should_continue_to_media(state: AdGenerationState) -> str:
    """Decide whether to continue to media generation or end."""
    if state["approve_concept"]:
        return "generate_video"
    else:
        return END


def create_concept_workflow(_config: Config) -> StateGraph:
    """Create workflow that stops at concept generation for human review."""

    workflow = StateGraph(AdGenerationState)

    # Add nodes for concept generation only
    workflow.add_node("scrape_web_content", scrape_web_content_node)
    workflow.add_node(
        "generate_business_description", generate_business_description_node
    )
    workflow.add_node("generate_concept", generate_concept_node)

    # Set entry point and routing
    workflow.set_entry_point("route_input")
    workflow.add_node("route_input", lambda state: state)  # Dummy node for routing

    # Route based on input type
    workflow.add_conditional_edges(
        "route_input",
        should_scrape_web_content,
        {"scrape_web": "scrape_web_content", "generate_concept": "generate_concept"},
    )

    # Web scraping path
    workflow.add_conditional_edges(
        "scrape_web_content",
        should_generate_business_description,
        {
            "generate_business_description": "generate_business_description",
            "generate_concept": "generate_concept",
        },
    )

    # Business description generation to concept
    workflow.add_edge("generate_business_description", "generate_concept")

    # End at concept generation
    workflow.add_edge("generate_concept", END)

    return workflow.compile()


def create_media_workflow(_config: Config) -> StateGraph:
    """Create workflow for script, visual plan, and media generation."""

    workflow = StateGraph(AdGenerationState)

    # Add media generation nodes
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("generate_visual_plan", generate_visual_plan_node)
    workflow.add_node("generate_video", generate_video_node)
    workflow.add_node("compose_video", compose_video_node)

    # Set entry point and flow
    workflow.set_entry_point("generate_script")
    workflow.add_edge("generate_script", "generate_visual_plan")
    workflow.add_edge("generate_visual_plan", "generate_video")
    workflow.add_edge("generate_video", "compose_video")
    workflow.add_edge("compose_video", END)

    return workflow.compile()
