# main.py
import asyncio
import json
import os
import re
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# CLI + terminal UI
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ADK runtime pieces
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

# Import the pipeline from the agent module
from political_profiles_agent.agent import root_agent


# Load environment variables (e.g., Gemini API key).
load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

APP_NAME = "PoliticalProfileCLI"
session_service = InMemorySessionService()

# Map agent names to user‑friendly stage labels for spinner updates.
STAGE_LABELS = {
    "DisambiguatePerson": "Disambiguating entity",
    "GovSources": "Collecting govt sources",
    "EncyclopediaSources": "Collecting encyclopedia/bio",
    "RecentUpdates": "Fetching recent updates",
    "ParallelResearch": "Running parallel research",
    "ConsolidateNotes": "Consolidating notes",
    "ExtractProfile": "Extracting structured profile",
    "ValidateProfile": "Validating profile",
    "PoliticalProfilePipeline": "Running profile pipeline",
}

def render_profile(profile: Dict[str, Any]) -> None:
    """Pretty print the final validated profile JSON."""
    title = profile.get("title", "").strip()
    bio = profile.get("biography", "").strip()
    current = profile.get("current_status", "").strip()

    # Title header
    title_text = Text(title or "Profile", style="bold green")

    # A simple two‑row table for Title and Current Status
    meta_tbl = Table.grid(padding=(0, 1))
    meta_tbl.add_row(Text("Title:", style="bold cyan"), Text(title or "—"))
    meta_tbl.add_row(Text("Current Status:", style="bold cyan"), Text(current or "—"))

    # Biography as a wrapped panel
    bio_panel = Panel.fit(Text(bio or "—"), title="Biography", title_align="left", border_style="blue")

    console.print(Panel.fit(meta_tbl, title=title_text, border_style="green"))
    console.print(bio_panel)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first top-level JSON object or array from a string, even if surrounded by ```
    Returns a dict/list on success, otherwise None.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Strip common triple-backtick code fences with or without language tags
    cleaned = re.sub(
        r"^\s*```(?:json|jsonc|json5)?\s*|\s*```",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Try a direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback: find the first JSON object or array in mixed content
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if not match:
        return None
    frag = match.group(1)
    try:
        return json.loads(frag)
    except Exception:
        return None


async def run_pipeline(query: str) -> Optional[Dict[str, Any]]:
    """
    Run the ADK pipeline with streaming events to drive a simple spinner UI,
    then return the final validated profile dict from session state.
    """
    # Create a session and runner
    session_id = f"session-{abs(hash(query))}"
    user_id = "default_user"
    await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    # One-line spinner with evolving stage text
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,  # clears line when done
        console=console,
    ) as progress:
        task_id = progress.add_task(description="Starting…", total=None)

        try:
            # Create a proper Content object for the message
            message_content = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )
            
            # Stream events; update spinner text when we see known authors/stages
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message_content
            ):
                author = getattr(event, "author", None) or ""
                if author in STAGE_LABELS:
                    progress.update(task_id, description=STAGE_LABELS[author])
                # If tools or sub‑agents emit granular authors, surface the most recent meaningful one
                elif author:
                    progress.update(task_id, description=f"Working: {author}")
        except Exception as e:
            progress.update(task_id, description="Error")
            console.print(f"[red]Run failed:[/red] {e}")
            import traceback
            console.print(f"[red]Traceback:[/red]\n{traceback.format_exc()}")
            return None

    # After run completes, the session state should contain the validated output under output_key
    session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    state = getattr(session, "state", {}) or {}
    
    final_profile = state.get("final_profile") or state.get("structured_profile") or state.get("fact_sheet")
    
    if not final_profile:
        console.print(f"[red]No profile found in state keys:[/red] {list(state.keys())}")
        return None
    
    # Handle different types of profile data
    if isinstance(final_profile, dict):
        # Already a dict, use it directly
        return final_profile
    elif isinstance(final_profile, str):
        parsed = _extract_json(final_profile)
        if parsed is not None:
            # If it's a list (unexpected), wrap or coerce into the expected dict shape
            if isinstance(parsed, list):
                return {"title": "Profile", "biography": json.dumps(parsed, ensure_ascii=False, indent=2), "current_status": "Unknown"}
            if isinstance(parsed, dict):
                return parsed
        console.print(f"[yellow]Profile is plain text, wrapping in dict[/yellow]")
        return {"biography": final_profile, "title": "Profile", "current_status": "Unknown"}
    elif hasattr(final_profile, 'model_dump'):
        # Pydantic model - convert to dict
        return final_profile.model_dump()
    elif hasattr(final_profile, 'dict'):
        # Older Pydantic model - convert to dict
        return final_profile.dict()
    else:
        # Try to convert to dict
        try:
            return dict(final_profile)
        except:
            console.print(f"[red]Cannot convert profile to dict, type:[/red] {type(final_profile)}")
            return None


@app.command(help="Build a concise, validated profile of an Indian politician.")
def main(
    name: str = typer.Option(None, "--name", "-n", help="Politician name (if omitted, will prompt)"),
):
    # Prompt interactively if not provided as an arg
    if not name:
        name = typer.prompt("Enter the politician's name")

    # Run the pipeline asynchronously
    profile = asyncio.run(run_pipeline(name))
    if not profile:
        console.print("[red]No profile produced by the pipeline.[/red]")
        raise typer.Exit(code=1)

    # Render the final JSON in a presentable form
    render_profile(profile)


if __name__ == "__main__":
    app()
