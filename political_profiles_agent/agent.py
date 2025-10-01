import os
from dotenv import load_dotenv

from typing import AsyncGenerator
from google.adk.tools import google_search
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.genai import types

load_dotenv()
from schemas.disambiguation import DisambiguationResult
from schemas.profile import ProfileOutput

disambiguate = LlmAgent(
    name="DisambiguatePerson",
    model="gemini-2.5-flash",
    instruction=(
        "Given a name, determine if it is an Indian politician. "
        "If not a politician, set is_politician=false and explain briefly in notes. "
        "If a politician, set is_politician=true, set normalized_name, entity_type='politician'. "
        "Return only the structured result."
    ),
    # tools=[google_search],
    output_schema=DisambiguationResult,
    output_key="entity_grounding",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=128,
        response_mime_type="application/json",
    ),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


gov_sources = LlmAgent(
    name="GovSources",
    model="gemini-2.5-flash",
    instruction=(
        "Using Google Search, find current official role, ministry/house membership, and portfolio from "
        "government and parliament portals only; produce a concise, citation-rich note."
    ),
    tools=[google_search],
    output_key="gov_note"
)

encyclopedia_sources = LlmAgent(
    name="EncyclopediaSources",
    model="gemini-2.5-flash",
    instruction=(
        "Using Google Search, gather education, career timeline, notable achievements from authoritative "
        "encyclopedias and official bios; produce a concise, citation-rich note."
    ),
    tools=[google_search],
    output_key="encyc_note"
)
# Recent updates: any role changes in last 90 days
recent_updates = LlmAgent(
    name="RecentUpdates",
    model="gemini-2.5-flash",
    instruction=(
        "Using Google Search, capture any role changes or major updates in the last 90 days with sources; "
        "produce a concise, citation-rich note."
    ),
    tools=[google_search],
    output_key="recent_note"
)
# Research in parallel: gov, encyclopedia, recent
parallel_research = ParallelAgent(
    name="ParallelResearch",
    sub_agents=[gov_sources, encyclopedia_sources, recent_updates],
)

#Consolidate: merge parallel notes into a single fact sheet
consolidate = LlmAgent(
    name="ConsolidateNotes",
    model="gemini-2.5-pro",
    instruction=(
        "Synthesize the gov_note, encyc_note, and recent_note into a single, internally consistent fact sheet, "
        "resolving conflicts by preferring official/government sources and the most recent authoritative updates; "
        "include inline source attributions in brackets."
    ),
    output_key="fact_sheet"
)

# Extract structured profile from the consolidated fact sheet
extract_structured = LlmAgent(
    name="ExtractProfile",
    model="gemini-2.5-flash",
    instruction=(
        "Use fact_sheet to produce ONLY a JSON object with exactly these keys: "
        "title, biography, current_status. "
        "title: If the politician holds any post now, output only the current office title without years "
        "(e.g., 'Prime Minister of India', 'Leader of the Opposition in Lok Sabha', 'Chief Minister of Uttar Pradesh'). "
        "Only if no current post exists, set title to 'Former highest role (years)' including service years if known. "
        "biography: 8–12 sentences including education, career, and achievements. "
        "current_status: clearly state present roles and responsibilities or 'Not in office' with the latest update. "
        "Return ONLY valid JSON, no markdown, no commentary."
    ),
    output_schema=ProfileOutput,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        max_output_tokens=2000,
    ),
    output_key="structured_profile",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# Validate and correct the extracted profile
validator = LlmAgent(
    name="ValidateProfile",
    model="gemini-2.5-flash",
    instruction=(
        "Validate and correct structured_profile using fact_sheet and prior notes so that: "
        "1) If a current post exists, title is only the current office (no years); "
        "2) Only when no current post exists, title is 'Former highest role (years)'; "
        "3) biography remains 8–12 sentences; "
        "4) current_status reflects the latest government/parliament sources. "
        "Return ONLY a JSON object with keys: title, biography, current_status."
    ),
    output_schema=ProfileOutput,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        max_output_tokens=2000,
    ),
    output_key="final_profile",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# friendly quick message agent when the name entered is not a politician
not_a_politician = LlmAgent(
    name="NotAPolitician",
    model="gemini-2.5-flash",
    instruction=(
        "Write a short, friendly message saying the entered name does not appear to be an Indian politician. "
        "Suggest entering the name of a political figure (e.g., an MP/MLA, Chief Minister, or Union Minister). "
        "Keep it under 3 sentences, no jargon."
    ),
    output_key="final_message",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=96,
    ),
)


# research → consolidate → extract → validate pipeline
research_pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[parallel_research, consolidate, extract_structured, validator],
    description="Research → Consolidate → Structure → Validate",
)
# Router agent: disambiguate → (not_a_politician | research_pipeline)
class Router(BaseAgent):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            description="Routes quickly to a friendly message if not a politician; else runs the research pipeline.",
            sub_agents=[disambiguate, not_a_politician, research_pipeline],
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # 1) Run fast disambiguation
        async for event in disambiguate.run_async(ctx):
            yield event

        gate = ctx.session.state.get("entity_grounding") or {}
        is_pol = bool(gate.get("is_politician"))

        # 2) Quick exit with friendly message if not a politician
        if not is_pol:
            async for event in not_a_politician.run_async(ctx):
                yield event
            return

        # 3) Otherwise, proceed with the full research pipeline
        async for event in research_pipeline.run_async(ctx):
            yield event


# Final root
root_agent = Router(name="PoliticalProfileRouter")
