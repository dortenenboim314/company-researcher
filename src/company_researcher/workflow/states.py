# src/company_researcher/models.py

from datetime import date
from typing import List, Optional
from pydantic import BaseModel, AnyUrl, Field


class InputState(BaseModel):
    company_name: str = Field(..., description="Full name of the company")
    company_url: AnyUrl = Field(..., description="Official website URL of the company")

class Founded(BaseModel):
    at: Optional[date] = Field(None, description="Founding date of the company", example="2020-10-15")
    by: Optional[List[str]] = Field(
        None, description="Names of the founders", examples=["Alice Smith", "Bob Johnson"]
    )

class CompanyBackground(BaseModel):
    industry: Optional[str] = Field(None, description="Primary industry or sector")
    founded: Optional[Founded] = Field(None, description="Founding details")
    description: Optional[str] = Field(
        None, description="Short narrative about the company's mission/vision"
    )
    key_milestones: Optional[List[str]] = Field(
        None, description="Notable milestones or achievements"
    )
    current_status: Optional[str] = Field(
        None, description="Brief statement of where the company stands today"
    )


class FinancialHealth(BaseModel):
    revenue: Optional[float] = Field(
        None,
        description="Latest annual revenue",
        example=1_000_000.0
    )
    funding_rounds: Optional[List[str]] = Field(
        default=None,
        description="List of funding rounds and amounts",
        example=[
            "Seed: $1M",
            "Series A: $5M by Acme Ventures",
            "Series B: $10M"
        ]
    )
    burn_rate: Optional[str] = Field(
        None,
        description="Estimated monthly burn rate",
        example="$200K/mo"
    )
    runway_months: Optional[float] = Field(
        None,
        description="Estimated runway in months",
        example=12.5
    )


class MarketPosition(BaseModel):
    competitors: Optional[List[str]] = Field(
        None, description="Key competitors"
    )
    unique_selling_points: Optional[List[str]] = Field(
        None, description="What sets the company apart"
    )
    market_share: Optional[str] = Field(
        None, description="Approximate market share or position"
    )


class NewsItem(BaseModel):
    title: str = Field(..., description="Headline of the news item")
    url: str = Field(..., description="Link to the news article")
    date_published: Optional[date] = Field(None, description="Publication date")


class ResearchState(BaseModel):
    company_name: str = Field(..., description="Full company name")
    company_url: str = Field(..., description="Official website URL")
    
    background: Optional[CompanyBackground] = Field(
        None, description="Structured background information"
    )
    financial_health: Optional[FinancialHealth] = Field(
        None, description="Structured financial health data"
    )
    market_position: Optional[MarketPosition] = Field(
        None, description="Structured market position insights"
    )
    
    recent_important_news: Optional[List[NewsItem]] = Field(
        default_factory=list, description="List of recent important news items"
    )
    final_report: Optional[str] = Field(
        None, description="Free-form consolidated report"
    )
    
    current_step: Optional[str] = Field(
        None, description="Which agent/step is running now"
    )
    errors: Optional[List[str]] = Field(
        default_factory=list, description="Any errors encountered"
    )