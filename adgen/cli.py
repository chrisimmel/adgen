import asyncio
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from adgen.models.ad import AdProject
from adgen.utils.checkpoint import WorkflowResumption
from adgen.utils.config import load_config
from adgen.utils.markdown import (
    generate_concept_markdown,
    generate_full_project_markdown,
    generate_script_markdown,
    generate_visual_plan_markdown,
    save_markdown,
)
from adgen.workflows.ad_generation import (
    AdGenerationState,
    checkpoint_manager,
    create_concept_workflow,
    create_media_workflow,
)

console = Console()


def display_markdown(content: str, title: str) -> None:
    """Display markdown content in a rich panel."""
    md = Markdown(content)
    panel = Panel(md, title=title, border_style="blue")
    console.print(panel)


def review_concept(project: AdProject) -> bool:
    """Display concept for human review and get approval."""
    if not project.concept:
        console.print("[red]No concept generated yet.[/red]")
        return False

    # Generate and display markdown
    concept_md = generate_concept_markdown(project.concept, project.project_id)
    display_markdown(concept_md, "📝 Generated Ad Concept")

    # Save to file
    save_path = save_markdown(concept_md, f"{project.project_id}_concept.md", "outputs")
    console.print(f"[green]Concept saved to: {save_path}[/green]")

    # Get user approval
    return Confirm.ask("Do you approve this concept?", default=True)


def review_script_and_plan(project: AdProject) -> bool:
    """Display script and visual plan for review."""
    if project.script:
        script_md = generate_script_markdown(project.script, project.project_id)
        display_markdown(script_md, "🎬 Generated Ad Script")

        save_path = save_markdown(
            script_md, f"{project.project_id}_script.md", "outputs"
        )
        console.print(f"[green]Script saved to: {save_path}[/green]")

    if project.visual_plan:
        plan_md = generate_visual_plan_markdown(project.visual_plan, project.project_id)
        display_markdown(plan_md, "🎨 Visual Plan")

        save_path = save_markdown(
            plan_md, f"{project.project_id}_visual_plan.md", "outputs"
        )
        console.print(f"[green]Visual plan saved to: {save_path}[/green]")

    # Save complete project summary
    full_md = generate_full_project_markdown(project)
    save_path = save_markdown(full_md, f"{project.project_id}_complete.md", "outputs")
    console.print(f"[green]Complete project summary saved to: {save_path}[/green]")

    return True


async def run_workflow(
    source_url: str | None = None,
    business_description: str | None = None,
    config_path: str | None = None,
) -> None:
    """Run the ad generation workflow."""

    # Validate input
    if not source_url and not business_description:
        console.print(
            "[red]Error: Either --url or --business-description must be provided[/red]"
        )
        return

    if source_url and business_description:
        console.print(
            "[red]Error: Provide either --url or --business-description, not both[/red]"
        )
        return

    # Load configuration
    try:
        config = load_config(Path(config_path) if config_path else None)
    except FileNotFoundError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        return
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return

    # Create project
    project_id = f"ad_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project = AdProject(
        source_url=source_url,
        business_description=business_description,
        project_id=project_id,
        created_at=datetime.now().isoformat(),
        status="created",
    )

    console.print(f"[blue]Starting ad generation for project: {project_id}[/blue]")
    if source_url:
        console.print(f"[dim]Source URL: {source_url}[/dim]")
    else:
        console.print(f"[dim]Business: {business_description}[/dim]")

    # Initialize state
    state = AdGenerationState(
        project=project, config=config, approve_concept=False, approve_final=False
    )

    try:
        # Phase 1: Run workflow up to concept review
        if source_url:
            console.print("[yellow]Scraping website content...[/yellow]")
        console.print("[yellow]Generating ad concept...[/yellow]")

        # Create a workflow that stops at concept review
        concept_workflow = create_concept_workflow(config)
        result = await concept_workflow.ainvoke(state)

        # Review concept
        if config.review.get("concept_approval", True):
            approved = review_concept(result["project"])
            if not approved:
                console.print("[red]Concept not approved. Exiting.[/red]")
                return
        else:
            approved = True

        # Phase 2: Continue with script, visual plan, and media generation if approved
        if approved:
            console.print(
                "[yellow]Generating script, visual plan, and media...[/yellow]"
            )
            result["approve_concept"] = True

            # Continue the workflow from where we left off
            media_workflow = create_media_workflow(config)
            result = await media_workflow.ainvoke(result)

            # Check if video and audio were generated
            if result["project"].assets:
                assets = result["project"].assets
                if assets.final_video_path:
                    console.print(
                        f"[green]✅ Final video composed: {assets.final_video_path}[/green]"
                    )
                elif assets.scene_clips:
                    console.print(
                        f"[green]✅ {len(assets.scene_clips)} scene clips generated[/green]"
                    )
                elif assets.video_path:
                    console.print(
                        f"[green]✅ Video generated: {assets.video_path}[/green]"
                    )

                if assets.audio_path:
                    console.print(
                        f"[green]✅ Voice-over audio generated: {assets.audio_path}[/green]"
                    )

            if result["project"].status in [
                "video_generation_failed",
                "video_composition_failed",
            ]:
                console.print(
                    "[yellow]⚠️ Video generation failed, but other components completed[/yellow]"
                )

            review_script_and_plan(result["project"])

            console.print("[green]✅ Ad generation workflow complete![/green]")
            console.print("[blue]Project files saved in: outputs/concepts/[/blue]")
            if result["project"].assets:
                assets = result["project"].assets
                if assets.final_video_path:
                    console.print(
                        f"[blue]Final composed video: {assets.final_video_path}[/blue]"
                    )
                elif assets.scene_clips:
                    console.print(
                        f"[blue]Scene clips ({len(assets.scene_clips)}): {[str(p) for p in assets.scene_clips]}[/blue]"
                    )
                elif assets.video_path:
                    console.print(f"[blue]Generated video: {assets.video_path}[/blue]")

                if assets.audio_path:
                    console.print(f"[blue]Voice-over audio: {assets.audio_path}[/blue]")

    except Exception as e:
        console.print(f"[red]Error during workflow execution: {e}[/red]")
        console.print("[dim]Check your API keys and configuration.[/dim]")


def list_checkpoints() -> None:
    """List all available checkpoints."""
    console.print("[bold blue]📋 Available Checkpoints[/bold blue]")
    console.print()

    checkpoints = checkpoint_manager.list_checkpoints()

    if not checkpoints:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    for checkpoint in checkpoints:
        console.print(f"[bold]{checkpoint['name']}[/bold]")
        console.print(f"  Project ID: {checkpoint['project_id']}")
        console.print(f"  Status: {checkpoint['status']}")
        console.print(f"  Timestamp: {checkpoint['timestamp']}")

        # Load checkpoint to show detailed information
        try:
            state = checkpoint_manager.load_checkpoint(checkpoint["name"])
            if state and state.get("project"):
                project = state["project"]

                # Show input source
                if project.source_url:
                    console.print(f"  Input: [blue]-u {project.source_url}[/blue]")
                elif project.business_description:
                    desc_preview = project.business_description[:60]
                    if len(project.business_description) > 60:
                        desc_preview += "..."
                    console.print(f'  Input: [blue]-b "{desc_preview}"[/blue]')

                # Extract brand/business name and concept preview
                if project.concept:
                    concept = project.concept

                    # Try to extract brand name from key message or target audience
                    brand_name = "Unknown"
                    if concept.key_message:
                        # Look for brand names in key message (capitalized words)
                        words = concept.key_message.split()
                        for word in words[:5]:  # Check first few words
                            if word[0].isupper() and len(word) > 2 and word.isalpha():
                                brand_name = word
                                break

                    console.print(f"  Brand: [green]{brand_name}[/green]")

                    if concept.target_audience:
                        audience_preview = concept.target_audience[:50]
                        if len(concept.target_audience) > 50:
                            audience_preview += "..."
                        console.print(f"  Audience: {audience_preview}")

                # Show video provider used
                if state.get("config"):
                    video_provider = state["config"].providers.get("video", "unknown")
                    console.print(
                        f"  Video Provider: [yellow]{video_provider}[/yellow]"
                    )

                # Show detailed asset information
                if project.assets:
                    assets = project.assets
                    console.print("  Assets:")

                    # Scene clips details
                    if assets.scene_clips:
                        valid_clips = 0
                        total_size = 0
                        for clip_path in assets.scene_clips:
                            if clip_path.exists() and clip_path.stat().st_size > 0:
                                valid_clips += 1
                                total_size += clip_path.stat().st_size

                        size_mb = total_size / (1024 * 1024)
                        if valid_clips > 0:
                            console.print(
                                f"    • Scene clips: [green]{valid_clips}/{len(assets.scene_clips)} valid[/green] ({size_mb:.1f}MB) [blue]← Can compose[/blue]"
                            )
                        else:
                            console.print(
                                f"    • Scene clips: [red]{valid_clips}/{len(assets.scene_clips)} valid[/red] ({size_mb:.1f}MB)"
                            )

                        # Show if they're fallback files
                        if assets.scene_clips and "fallback" in str(
                            assets.scene_clips[0]
                        ):
                            console.print(
                                "    • [red]⚠️  Contains fallback/mock videos[/red]"
                            )

                    # Final video details
                    if assets.final_video_path:
                        if assets.final_video_path.exists():
                            size_mb = assets.final_video_path.stat().st_size / (
                                1024 * 1024
                            )
                            if assets.scene_clips:
                                console.print(
                                    f"    • Final video: [green]✓[/green] ({size_mb:.1f}MB) [dim]← Pre-composed[/dim]"
                                )
                            else:
                                console.print(
                                    f"    • Final video: [green]✓[/green] ({size_mb:.1f}MB) [blue]← Ready for audio overlay[/blue]"
                                )
                        else:
                            console.print("    • Final video: [red]✗ Missing[/red]")

                    # Audio details
                    if assets.audio_path:
                        if assets.audio_path.exists():
                            size_kb = assets.audio_path.stat().st_size / 1024
                            console.print(
                                f"    • Audio: [green]✓[/green] ({size_kb:.0f}KB)"
                            )
                        else:
                            console.print("    • Audio: [red]✗ Missing[/red]")

                # Show failure reasons for failed statuses
                if "failed" in checkpoint["status"]:
                    console.print(
                        f"  [red]Failure: {checkpoint['status'].replace('_', ' ').title()}[/red]"
                    )

        except Exception as e:
            console.print(f"  [dim]Error loading checkpoint details: {e}[/dim]")

        console.print()


async def resume_workflow(
    checkpoint_name: str, config_path: str, restart_from: str | None = None
) -> None:
    """Resume workflow from a checkpoint, optionally restarting from a specific step."""
    from pathlib import Path
    import time

    from moviepy import CompositeAudioClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.io.VideoFileClip import VideoFileClip

    if restart_from:
        console.print(
            f"[blue]🔄 Restarting workflow from step '{restart_from}' using checkpoint: {checkpoint_name}[/blue]"
        )
    else:
        console.print(
            f"[blue]🔄 Resuming workflow from checkpoint: {checkpoint_name}[/blue]"
        )

    resumption = WorkflowResumption(checkpoint_manager)
    resume_result = resumption.resume_workflow(checkpoint_name)

    if not resume_result:
        console.print("[red]Failed to resume workflow[/red]")
        return

    state, next_step = resume_result

    # Load configuration
    try:
        config = load_config(Path(config_path) if config_path else None)
        # Update state with current config (in case config changed)
        state["config"] = config

        # Convert to AdGenerationState
        state = AdGenerationState(state)
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        return

    console.print("[green]✅ Checkpoint loaded successfully[/green]")
    console.print(f"[dim]Project: {state['project'].project_id}[/dim]")
    console.print(f"[dim]Current status: {state['project'].status}[/dim]")
    console.print(f"[dim]Next step: {next_step}[/dim]")

    # Override next step if restarting from specific step
    if restart_from:
        step_mapping = {
            "concept": "concept_workflow",
            "script": "media_workflow",
            "visual_plan": "media_workflow",
            "video": "generate_video",
            "audio": "generate_audio",
            "compose": "compose_video",
        }
        next_step = step_mapping.get(restart_from, next_step)
        console.print(
            f"[yellow]🔄 Overriding to restart from: {restart_from} (step: {next_step})[/yellow]"
        )

        # Reset project status to trigger regeneration
        if restart_from == "audio":
            state["project"].status = "video_composed"
            # Remove existing audio to force regeneration
            if state["project"].assets and state["project"].assets.audio_path:
                console.print("[dim]Removing existing audio for regeneration[/dim]")
                state["project"].assets.audio_path = None
        elif restart_from == "compose":
            state["project"].status = "audio_generated"
        elif restart_from == "video":
            state["project"].status = "visual_plan_generated"
            # Remove existing video assets
            if state["project"].assets:
                state["project"].assets.video_path = None
                state["project"].assets.scene_clips = None
                state["project"].assets.final_video_path = None

    try:
        if next_step == "concept_workflow":
            # Restart from concept generation
            console.print("[yellow]Regenerating concept...[/yellow]")
            concept_workflow = create_concept_workflow(config)
            result = await concept_workflow.ainvoke(state)

            # Review concept
            if config.review.get("concept_approval", True):
                approved = review_concept(result["project"])
                if not approved:
                    console.print("[red]Concept not approved. Exiting.[/red]")
                    return
            else:
                approved = True

            if approved:
                result["approve_concept"] = True
                console.print("[yellow]Continuing with media generation...[/yellow]")
                media_workflow = create_media_workflow(config)
                result = await media_workflow.ainvoke(result)
                review_script_and_plan(result["project"])
                console.print("[green]✅ Restarted workflow complete![/green]")

        elif next_step == "generate_video":
            console.print("[yellow]Regenerating video...[/yellow]")
            from adgen.workflows.ad_generation import generate_video_node

            state["approve_concept"] = True
            result = await generate_video_node(state)

            console.print("[yellow]Continuing with audio generation...[/yellow]")
            from adgen.workflows.ad_generation import generate_audio_node

            result = await generate_audio_node(result)

            # Check if we need video composition or audio overlay on existing video
            if (
                result["project"].assets
                and result["project"].assets.final_video_path
                and not result["project"].assets.scene_clips
            ):
                console.print(
                    "[yellow]Applying audio overlay to existing final video...[/yellow]"
                )
                # Apply audio directly to existing final video using smart audio logic

                try:
                    final_video_path = result["project"].assets.final_video_path
                    audio_path = result["project"].assets.audio_path

                    if final_video_path.exists() and audio_path.exists():
                        # Load video and audio
                        video_clip = VideoFileClip(str(final_video_path))
                        audio_clip = AudioFileClip(str(audio_path))

                        # Trim audio to match video duration
                        if audio_clip.duration > video_clip.duration:
                            console.print(
                                f"Trimming audio from {audio_clip.duration:.1f}s to {video_clip.duration:.1f}s"
                            )
                            audio_clip = audio_clip.subclipped(0, video_clip.duration)

                        # Smart audio handling: Check if video provider generates audio
                        video_provider = result["config"].providers.get(
                            "video", "runwayml"
                        )
                        video_config = result["config"].video.get(video_provider, {})
                        generates_audio = video_config.get("generate_audio", False)

                        if generates_audio and video_clip.audio is not None:
                            # Video has original audio (e.g., Veo 3) - create overlay
                            console.print(
                                "Creating audio overlay (original + voice-over)"
                            )
                            original_audio = video_clip.audio
                            original_reduced = original_audio.with_volume_scaled(0.3)
                            composite_audio = CompositeAudioClip(
                                [original_reduced, audio_clip]
                            )
                            final_clip = video_clip.with_audio(composite_audio)
                        else:
                            # No original audio worth preserving (e.g., Runway) - replace
                            console.print("Replacing audio with voice-over")
                            final_clip = video_clip.with_audio(audio_clip)

                        # Create new output path
                        timestamp = int(time.time())
                        project_id = result["project"].project_id
                        new_video_path = Path(
                            f"outputs/media/{project_id}_final_with_new_audio_{timestamp}.mp4"
                        )

                        # Write new video
                        console.print(
                            f"Writing video with new audio to: {new_video_path}"
                        )
                        final_clip.write_videofile(
                            str(new_video_path),
                            codec="libx264",
                            audio_codec="aac",
                            temp_audiofile="temp-audio.m4a",
                            remove_temp=True,
                            logger=None,
                        )

                        # Update assets with new video path
                        result["project"].assets.final_video_path = new_video_path
                        result["project"].status = "video_composed"

                        # Cleanup
                        video_clip.close()
                        audio_clip.close()
                        final_clip.close()

                        console.print(
                            "[green]✅ Audio successfully applied to existing video[/green]"
                        )
                        console.print(f"[blue]New video: {new_video_path}[/blue]")

                except Exception as e:
                    console.print(
                        f"[red]Failed to apply audio to existing video: {e}[/red]"
                    )
                    import traceback

                    traceback.print_exc()
            else:
                console.print("[yellow]Continuing with video composition...[/yellow]")
                from adgen.workflows.ad_generation import compose_video_node

                result = await compose_video_node(result)

            review_script_and_plan(result["project"])
            console.print("[green]✅ Restarted workflow complete![/green]")

        elif next_step == "media_workflow":
            # Need concept approval first
            if not review_concept(state["project"]):
                console.print("[red]Concept not approved. Workflow stopped.[/red]")
                return

            state["approve_concept"] = True

            # Run media workflow
            console.print("[yellow]Continuing with media generation...[/yellow]")
            media_workflow = create_media_workflow(config)
            result = await media_workflow.ainvoke(state)

            # Display results same as normal workflow
            review_script_and_plan(result["project"])

            console.print("[green]✅ Resumed workflow complete![/green]")

        elif next_step == "generate_audio":
            console.print("[yellow]Continuing with audio generation...[/yellow]")

            # Import and run audio generation node
            from adgen.workflows.ad_generation import generate_audio_node

            state["approve_concept"] = True  # Must be approved to reach this point
            result = await generate_audio_node(state)

            # Check if we need video composition or audio overlay on existing video
            if (
                result["project"].assets
                and result["project"].assets.final_video_path
                and not result["project"].assets.scene_clips
            ):
                console.print(
                    "[yellow]Applying audio overlay to existing final video...[/yellow]"
                )
                # Apply audio directly to existing final video using smart audio logic

                try:
                    final_video_path = result["project"].assets.final_video_path
                    audio_path = result["project"].assets.audio_path

                    if final_video_path.exists() and audio_path.exists():
                        # Load video and audio
                        video_clip = VideoFileClip(str(final_video_path))
                        audio_clip = AudioFileClip(str(audio_path))

                        # Trim audio to match video duration
                        if audio_clip.duration > video_clip.duration:
                            console.print(
                                f"Trimming audio from {audio_clip.duration:.1f}s to {video_clip.duration:.1f}s"
                            )
                            audio_clip = audio_clip.subclipped(0, video_clip.duration)

                        # Smart audio handling: Check if video provider generates audio
                        video_provider = result["config"].providers.get(
                            "video", "runwayml"
                        )
                        video_config = result["config"].video.get(video_provider, {})
                        generates_audio = video_config.get("generate_audio", False)

                        if generates_audio and video_clip.audio is not None:
                            # Video has original audio (e.g., Veo 3) - create overlay
                            console.print(
                                "Creating audio overlay (original + voice-over)"
                            )
                            original_audio = video_clip.audio
                            original_reduced = original_audio.with_volume_scaled(0.3)
                            composite_audio = CompositeAudioClip(
                                [original_reduced, audio_clip]
                            )
                            final_clip = video_clip.with_audio(composite_audio)
                        else:
                            # No original audio worth preserving (e.g., Runway) - replace
                            console.print("Replacing audio with voice-over")
                            final_clip = video_clip.with_audio(audio_clip)

                        # Create new output path
                        timestamp = int(time.time())
                        project_id = result["project"].project_id
                        new_video_path = Path(
                            f"outputs/media/{project_id}_final_with_new_audio_{timestamp}.mp4"
                        )

                        # Write new video
                        console.print(
                            f"Writing video with new audio to: {new_video_path}"
                        )
                        final_clip.write_videofile(
                            str(new_video_path),
                            codec="libx264",
                            audio_codec="aac",
                            temp_audiofile="temp-audio.m4a",
                            remove_temp=True,
                            logger=None,
                        )

                        # Update assets with new video path
                        result["project"].assets.final_video_path = new_video_path
                        result["project"].status = "video_composed"

                        # Cleanup
                        video_clip.close()
                        audio_clip.close()
                        final_clip.close()

                        console.print(
                            "[green]✅ Audio successfully applied to existing video[/green]"
                        )
                        console.print(f"[blue]New video: {new_video_path}[/blue]")

                except Exception as e:
                    console.print(
                        f"[red]Failed to apply audio to existing video: {e}[/red]"
                    )
                    import traceback

                    traceback.print_exc()
            else:
                console.print("[yellow]Continuing with video composition...[/yellow]")

                # Import and run composition node
                from adgen.workflows.ad_generation import compose_video_node

                result = await compose_video_node(result)

            # Display results
            review_script_and_plan(result["project"])

            console.print("[green]✅ Resumed workflow complete![/green]")

        elif next_step == "compose_video":
            console.print("[yellow]Continuing with video composition...[/yellow]")

            # Import and run composition node
            from adgen.workflows.ad_generation import compose_video_node

            state["approve_concept"] = True  # Must be approved to reach this point
            result = await compose_video_node(state)

            # Display results
            review_script_and_plan(result["project"])

            console.print("[green]✅ Resumed workflow complete![/green]")

        elif next_step == "complete":
            console.print("[green]✅ Workflow is already completed![/green]")

            # Display the final results
            review_script_and_plan(state["project"])

            # Show final assets if available
            if state["project"].assets:
                assets = state["project"].assets
                if assets.final_video_path:
                    console.print(
                        f"[blue]Final video: {assets.final_video_path}[/blue]"
                    )
                elif assets.scene_clips:
                    console.print(
                        f"[blue]Scene clips: {len(assets.scene_clips)} files[/blue]"
                    )
                if assets.audio_path:
                    console.print(f"[blue]Audio: {assets.audio_path}[/blue]")

        else:
            console.print(
                f"[yellow]Resume from {next_step} not yet implemented[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error during resumed workflow: {e}[/red]")


@click.group()
def cli():
    """AdGen - AI Video Ad Generator"""
    pass


@cli.command()
@click.option(
    "--url",
    "-u",
    help="URL of the business website to analyze",
)
@click.option(
    "--business-description",
    "-b",
    help="Direct description of the business and product/service to advertise",
)
@click.option(
    "--config", "-c", help="Path to configuration file", default="config.yaml"
)
def generate(url: str | None, business_description: str | None, config: str) -> None:
    """Generate AI-powered video advertisements.

    Provide either a website URL to analyze or a direct business description.
    """

    console.print("[bold blue]🎥 AdGen - AI Video Ad Generator[/bold blue]")
    console.print()

    # If neither provided, prompt for one
    if not url and not business_description:
        console.print("Choose input method:")
        console.print("1. Website URL (analyzes website content)")
        console.print("2. Direct business description")
        choice = click.prompt("Enter choice (1 or 2)", type=click.Choice(["1", "2"]))

        if choice == "1":
            url = click.prompt("Enter website URL")
        else:
            business_description = click.prompt(
                "Describe your business and what you want to advertise"
            )

    # Run the async workflow
    asyncio.run(run_workflow(url, business_description, config))


@cli.command()
def checkpoints():
    """List all available workflow checkpoints."""
    list_checkpoints()


@cli.command()
@click.argument("checkpoint_name")
@click.option(
    "--config", "-c", help="Path to configuration file", default="config.yaml"
)
@click.option(
    "--restart-from",
    help="Restart from a specific step instead of resuming from next step",
    type=click.Choice(
        ["concept", "script", "visual_plan", "video", "audio", "compose"]
    ),
    default=None,
)
def resume(checkpoint_name: str, config: str, restart_from: str | None):
    """Resume workflow from a checkpoint, optionally restarting from a specific step."""
    asyncio.run(resume_workflow(checkpoint_name, config, restart_from))


@cli.command()
@click.argument("checkpoint_name")
def delete(checkpoint_name: str):
    """Delete a workflow checkpoint."""
    if checkpoint_manager.delete_checkpoint(checkpoint_name):
        console.print(f"[green]✅ Checkpoint '{checkpoint_name}' deleted[/green]")
    else:
        console.print(f"[red]❌ Checkpoint '{checkpoint_name}' not found[/red]")


if __name__ == "__main__":
    cli()
