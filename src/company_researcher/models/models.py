from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    company_name: str
    company_url: str

    
class NewsItem(BaseModel):
    title: str = Field(..., description="Headline of the news item")
    url: str = Field(..., description="Link to the news article")
    date_published: Optional[date] = Field(None, description="Publication date")
    
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

class News(BaseModel):
    recent_important_news: Optional[List[NewsItem]] = Field(
        default=None, description="List of recent important news items"
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

class FinalReport(BaseModel):
    background_summary: Optional[str] = Field(
        None, description="Summary of company background information"
    )
    financial_health_summary: Optional[str] = Field(
        None, description="Summary of financial health data"
    )
    market_position_summary: Optional[str] = Field(
        None, description="Summary of market position insights"
    )
    news_summary: Optional[str] = Field(
        None, description="Summary of recent news items"
    )
    

class ResearchResponse(BaseModel):
    background: Optional[CompanyBackground] = Field(
        None, description="Structured background information"
    )
    financial_health: Optional[FinancialHealth] = Field(
        None, description="Structured financial health data"
    )
    market_position: Optional[MarketPosition] = Field(
        None, description="Structured market position insights"
    )
    
    news: Optional[News] = Field(
        default=None, description="Structured recent news information"
    )
    final_report: Optional[FinalReport] = Field(
        None, description="Final research report summarizing all findings"
    )
