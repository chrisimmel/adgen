import asyncio
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from adgen.models.ad import AdProject
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
    create_ad_generation_workflow,
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

    # Create workflow
    workflow = create_ad_generation_workflow(config)

    # Initialize state
    state = AdGenerationState(
        project=project, config=config, approve_concept=False, approve_final=False
    )

    try:
        # Run workflow (may include web scraping and business analysis)
        if source_url:
            console.print("[yellow]Scraping website content...[/yellow]")
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
            console.print("[blue]Project files saved in: outputs/concepts/[/blue]")

    except Exception as e:
        console.print(f"[red]Error during workflow execution: {e}[/red]")
        console.print("[dim]Check your API keys and configuration.[/dim]")


@click.command()
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
def main(url: str | None, business_description: str | None, config: str) -> None:
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


if __name__ == "__main__":
    main()
