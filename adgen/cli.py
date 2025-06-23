from typing import Optional

import click
import asyncio
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from adgen.models.ad import AdProject
from adgen.workflows.ad_generation import (
    create_ad_generation_workflow,
    AdGenerationState,
)
from adgen.utils.config import load_config
from adgen.utils.markdown import (
    generate_concept_markdown,
    generate_script_markdown,
    generate_visual_plan_markdown,
    generate_full_project_markdown,
    save_markdown,
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
    business_description: str, config_path: Optional[str] = None
) -> None:
    """Run the ad generation workflow."""

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
        business_description=business_description,
        project_id=project_id,
        created_at=datetime.now().isoformat(),
        status="created",
    )

    console.print(f"[blue]Starting ad generation for project: {project_id}[/blue]")
    console.print(f"[dim]Business: {business_description}[/dim]")

    # Create workflow
    workflow = create_ad_generation_workflow(config)

    # Initialize state
    state = AdGenerationState(
        project=project, config=config, approve_concept=False, approve_final=False
    )

    try:
        # Run concept generation
        console.print("[yellow]Generating ad concept...[/yellow]")
        result = await workflow.ainvoke(state)

        # Review concept
        if config.review.get("concept_approval", True):
            approved = review_concept(result["project"])
            result["approve_concept"] = approved

            if not approved:
                console.print("[red]Concept not approved. Exiting.[/red]")
                return
        else:
            result["approve_concept"] = True

        # Continue with script and visual plan if approved
        if result["approve_concept"]:
            console.print("[yellow]Generating script and visual plan...[/yellow]")

            # Note: In a more complete implementation, we'd continue the workflow
            # For now, we'll just display what we have
            review_script_and_plan(result["project"])

            console.print("[green]✅ Ad concept and plan generation complete![/green]")
            console.print(f"[blue]Project files saved in: outputs/concepts/[/blue]")

    except Exception as e:
        console.print(f"[red]Error during workflow execution: {e}[/red]")
        console.print("[dim]Check your API keys and configuration.[/dim]")


@click.command()
@click.option(
    "--business-description",
    "-b",
    prompt="Describe your business and what you want to advertise",
    help="Description of the business and product/service to advertise",
)
@click.option(
    "--config", "-c", help="Path to configuration file", default="config.yaml"
)
def main(business_description: str, config: str) -> None:
    """Generate AI-powered video advertisements."""

    console.print("[bold blue]🎥 AdGen - AI Video Ad Generator[/bold blue]")
    console.print()

    # Run the async workflow
    asyncio.run(run_workflow(business_description, config))


if __name__ == "__main__":
    main()
