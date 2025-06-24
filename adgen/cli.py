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
        console.print()


async def resume_workflow(checkpoint_name: str, config_path: str) -> None:
    """Resume workflow from a checkpoint."""
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

    try:
        if next_step == "media_workflow":
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

        else:
            console.print(
                f"[yellow]Resume from {next_step} not yet implemented[/yellow]"
            )

    except Exception as e:
        console.print(f"[red]Error during resumed workflow: {e}[/red]")


@click.group(invoke_without_command=True)
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
@click.pass_context
def cli(
    ctx: click.Context, url: str | None, business_description: str | None, config: str
):
    """AdGen - AI Video Ad Generator"""
    # If no subcommand was invoked, run generate workflow directly (backward compatibility)
    if ctx.invoked_subcommand is None:
        # Backward compatibility: run generate workflow directly
        console.print("[bold blue]🎥 AdGen - AI Video Ad Generator[/bold blue]")
        console.print()

        # If neither provided, prompt for one
        if not url and not business_description:
            console.print("Choose input method:")
            console.print("1. Website URL (analyzes website content)")
            console.print("2. Direct business description")
            choice = click.prompt(
                "Enter choice (1 or 2)", type=click.Choice(["1", "2"])
            )

            if choice == "1":
                url = click.prompt("Enter website URL")
            else:
                business_description = click.prompt(
                    "Describe your business and what you want to advertise"
                )

        # Run the async workflow
        asyncio.run(run_workflow(url, business_description, config))


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
def resume(checkpoint_name: str, config: str):
    """Resume workflow from a checkpoint."""
    asyncio.run(resume_workflow(checkpoint_name, config))


@cli.command()
@click.argument("checkpoint_name")
def delete(checkpoint_name: str):
    """Delete a workflow checkpoint."""
    if checkpoint_manager.delete_checkpoint(checkpoint_name):
        console.print(f"[green]✅ Checkpoint '{checkpoint_name}' deleted[/green]")
    else:
        console.print(f"[red]❌ Checkpoint '{checkpoint_name}' not found[/red]")


# For backward compatibility, keep the old main function
def main(
    url: str | None = None,
    business_description: str | None = None,
    config: str = "config.yaml",
) -> None:
    """Generate AI-powered video advertisements."""
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


if __name__ == "__main__":
    cli()
