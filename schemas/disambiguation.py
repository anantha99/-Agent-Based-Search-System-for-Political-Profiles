from pydantic import BaseModel, Field


class DisambiguationResult(BaseModel):
    is_politician: bool = Field(description="True only if this person is an Indian politician")
    normalized_name: str = Field(description="Canonical name or empty if not applicable")
    entity_type: str = Field(description="e.g., 'politician', 'actor', 'businessperson', 'unknown'")
    notes: str = Field(description="One-line identity or why not a politician")