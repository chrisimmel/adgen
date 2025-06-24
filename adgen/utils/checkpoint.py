from datetime import datetime
import json
from pathlib import Path
from typing import Any


class CheckpointManager:
    """Manages workflow state checkpoints for resuming interrupted sessions."""

    def __init__(self, checkpoint_dir: str = "outputs/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state: dict[str, Any], checkpoint_name: str) -> Path:
        """Save workflow state to a checkpoint file."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.checkpoint"

        # Create checkpoint data
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "project_id": state["project"].project_id,
            "status": state["project"].status,
            "approve_concept": state.get("approve_concept", False),
            "approve_final": state.get("approve_final", False),
            "project": state["project"].model_dump(),  # Serialize Pydantic model
            "config": state["config"].model_dump(),
        }

        # Save as JSON for readability and portability
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_name: str) -> dict[str, Any] | None:
        """Load workflow state from a checkpoint file."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.checkpoint"

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            # Reconstruct AdProject from serialized data
            from adgen.models.ad import AdProject
            from adgen.utils.config import Config

            project = AdProject(**checkpoint_data["project"])
            config = Config(**checkpoint_data["config"])

            # Create state dict
            state = {
                "project": project,
                "config": config,
                "approve_concept": checkpoint_data.get("approve_concept", False),
                "approve_final": checkpoint_data.get("approve_final", False),
            }

            print(f"Checkpoint loaded: {checkpoint_path}")
            print(f"Project ID: {project.project_id}")
            print(f"Status: {project.status}")
            print(f"Timestamp: {checkpoint_data['timestamp']}")

            return state

        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.checkpoint"):
            try:
                with open(checkpoint_file, encoding="utf-8") as f:
                    data = json.load(f)

                checkpoints.append(
                    {
                        "name": checkpoint_file.stem,
                        "project_id": data.get("project_id"),
                        "status": data.get("status"),
                        "timestamp": data.get("timestamp"),
                        "path": checkpoint_file,
                    }
                )
            except Exception as e:
                print(f"Failed to read checkpoint {checkpoint_file}: {e}")

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a checkpoint file."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.checkpoint"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Checkpoint deleted: {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

    def auto_checkpoint_name(self, project_id: str, status: str) -> str:
        """Generate automatic checkpoint name based on project and status."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{project_id}_{status}_{timestamp}"


def create_checkpoint_decorator(checkpoint_manager: CheckpointManager):
    """Decorator factory to automatically checkpoint after workflow nodes."""

    def checkpoint_after(node_func):
        """Decorator that saves a checkpoint after a node completes successfully."""

        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            # Save the original status before executing the node
            original_status = state["project"].status

            # Execute the original node
            result_state = await node_func(state)

            # Save checkpoint if status changed (indicating progress)
            if result_state["project"].status != original_status:
                checkpoint_name = checkpoint_manager.auto_checkpoint_name(
                    result_state["project"].project_id, result_state["project"].status
                )
                checkpoint_manager.save_checkpoint(result_state, checkpoint_name)
                print(f"✅ Checkpoint saved: {checkpoint_name}")

            return result_state

        return wrapper

    return checkpoint_after


class WorkflowResumption:
    """Handles resuming workflows from checkpoints."""

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager

    def can_resume_from_status(self, status: str) -> bool:
        """Check if workflow can be resumed from a given status."""
        resumable_statuses = {
            "concept_generated",
            "script_generated",
            "visual_plan_generated",
            "scene_clips_generated",
            "audio_generated",
            "video_composed",
        }
        return status in resumable_statuses

    def get_next_workflow_step(self, status: str) -> str | None:
        """Determine the next workflow step based on current status."""
        status_to_next_step = {
            "concept_generated": "media_workflow",
            "script_generated": "generate_visual_plan",
            "visual_plan_generated": "generate_video",
            "scene_clips_generated": "generate_audio",
            "audio_generated": "compose_video",
            "video_composed": "complete",
        }
        return status_to_next_step.get(status)

    def resume_workflow(self, checkpoint_name: str):
        """Resume workflow from a checkpoint."""
        state = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        if not state:
            return None

        current_status = state["project"].status
        next_step = self.get_next_workflow_step(current_status)

        if not next_step:
            print(f"Cannot determine next step for status: {current_status}")
            return None

        if next_step == "complete":
            print("Workflow already completed!")
            return state

        print(f"Resuming workflow from: {current_status}")
        print(f"Next step: {next_step}")

        return state, next_step
