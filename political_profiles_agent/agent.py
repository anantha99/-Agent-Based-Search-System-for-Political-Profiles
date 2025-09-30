import os
from dotenv import load_dotenv

from google.adk.tools import google_search
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.genai import types

load_dotenv()


#Disambiguation: ensure the correct person/entity
disambiguate = LlmAgent(
    name="DisambiguatePerson",
    model="gemini-2.5-flash",
    instruction=(
        "Given a name of an Indian politician, identify the correct person, "
        "return a one-line identity statement and 3-5 canonical sources to use (official gov, election commission, "
        "parliament, and authoritative encyclopedic references)."
    ),
    tools=[google_search],
    output_key="entity_grounding"
)

#Parallel grounded research from multiple angles
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

#Structured extraction with strict JSON schema
class ProfileOutput(BaseModel):
    Title: str = Field(description=(
        "If the politician currently holds any political office, set to that current office "
        "(e.g., 'Prime Minister of India', 'Leader of the Opposition in Lok Sabha', 'Chief Minister of Uttar Pradesh') "
        "with no years included. Only if no current office is held, set to 'Former highest role (years)' with service "
        "years if known (e.g., 'Former Prime Minister (2004–2014)') or a single year if only the final year is known."
    ))
    Biography: str = Field(description="10-15 sentence biography")
    Current_Status: str = Field(alias="Current Status", description="Present role and responsibilities")


profile_schema = {
  "type": "OBJECT",
  "properties": {
    "Title": {"type": "STRING"},
    "Biography": {"type": "STRING"},
    "Current Status": {"type": "STRING"}
  },
  "required": ["Title", "Biography", "Current Status"]
}

extract_structured = LlmAgent(
    name="ExtractProfile",
    model="gemini-2.5-pro",
    instruction=(
        "Use fact_sheet to produce a concise JSON object with keys: Title, Biography, Current Status. "
        "Title: If the politician holds any post now, output only the current office title without years "
        "(e.g., 'Prime Minister of India', 'Leader of the Opposition in Lok Sabha', 'Chief Minister of Uttar Pradesh'). "
        "Only if they hold no current post, set Title to 'Former highest role (years)' including service years if known. "
        "Biography:10-15 sentences including education, career and achievements. Current Status: clearly state the present roles and responsibilities or 'Not in office' with any current latest update on them. "
        "Do not add extra keys. Return only valid JSON."
    ),
    output_schema=ProfileOutput, 
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2
    ),
    output_key="structured_profile"
)



#Lightweight validator
validator = LlmAgent(
    name="ValidateProfile",
    model="gemini-2.5-flash",
    instruction=(
        "Validate and correct structured_profile using fact_sheet and prior notes so that: "
        "1) If a current post exists, Title is only the current office title with no years; "
        "2) Only when no current post exists, Title is 'Former highest role (years)' with service years if available; "
        "3) Biography remains 10-15 sentences including education, career and achievements; "
        "4) Current Status reflects the latest government/parliament sources. "
        "If inconsistencies exist, correct them using the provided notes without changing schema."
    ),
    output_key="final_profile"
)



# Root pipeline: sequential orchestration
root_agent = SequentialAgent(
    name="PoliticalProfilePipeline",
    sub_agents=[disambiguate, parallel_research, consolidate, extract_structured, validator],
    description="Search → Summarize → Structure pipeline for politician profiles"
)
