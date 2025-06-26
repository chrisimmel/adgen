# Claude Preferences for AdGen Project

## Git Operations

- **User handles all Git operations** - Claude should NOT run git commands (add, commit, push, etc.)
- User prefers to test and examine all changes before committing
- Claude should focus on code implementation and let user handle version control

## Development Workflow

- Claude can suggest testing commands or validation steps
- User will handle dependency installation and testing
- User will manage commits and pull requests

## Communication Style

- Keep responses concise and focused on the technical implementation
- Avoid git-related suggestions unless specifically asked

## Python Dependencies

- Dependencies are managed with uv and pyproject.toml
- To run a script with correct dependencies, always use 'uv run python ...'.
