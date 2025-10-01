
from pydantic import BaseModel, Field

class ProfileOutput(BaseModel):
    title: str = Field(description=(
        "If the politician currently holds any political office, set to that current office "
        "(e.g., 'Prime Minister of India', 'Leader of the Opposition in Lok Sabha', 'Chief Minister of Uttar Pradesh') "
        "with no years included. Only if no current office is held, set to 'Former highest role (years)' with service "
        "years if known."
    ))
    biography: str = Field(description="8–12 sentence biography to reduce truncation risk")
    current_status: str = Field(description="Present role and responsibilities, or 'Not in office' with latest update")