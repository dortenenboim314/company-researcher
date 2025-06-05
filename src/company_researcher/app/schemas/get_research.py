from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field


class GetResearchRequest(BaseModel):
    company_name: str
    company_url: str

class GetResearchResponse(BaseModel):
    background_summary: Optional[str] = Field(
        None, description="Summary of company background information"
    )
    financial_health_summary: Optional[str] = Field(
        None, description="Summary of financial health data"
    )
    market_position_summary: Optional[str] = Field(
        None, description="Summary of market position insights"
    )
    positive_aspects: list[str] = Field(description="List of Positive aspects of the company, such as strengths, opportunities, and positive trends.")
    negative_aspects: list[str] = Field(description="List of Negative aspects of the company, such as weaknesses, threats, and negative trends.")
    
