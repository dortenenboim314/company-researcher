
from datetime import date
from company_researcher.models import (
    ResearchResponse,
    CompanyBackground,
    Founded,
    NewsItem,
    News,
    FinancialHealth,
    MarketPosition,
    FinalReport,
)


def mock_research_response() -> ResearchResponse:
    return ResearchResponse(
        background=CompanyBackground(
            industry="Artificial Intelligence",
            founded=Founded(
                at=date(2022, 4, 10),
                by=["Shalev Hulio", "Gil Dolev"]
            ),
            description="Tavily is an AI-powered research assistant that helps users retrieve and summarize information instantly.",
            key_milestones=[
                "Founded in 2022",
                "Launched beta in early 2023",
                "Raised Series A funding"
            ],
            current_status="Scaling operations and expanding customer base"
        ),
        financial_health=FinancialHealth(
            revenue=1_200_000.00,
            funding_rounds=[
                "Seed: $1M",
                "Series A: $5M by Acme Ventures"
            ],
            burn_rate="$200K/mo",
            runway_months=12.0
        ),
        market_position=MarketPosition(
            competitors=["Perplexity", "You.com"],
            unique_selling_points=[
                "LLM-native research orchestration",
                "Faster, more relevant results"
            ],
            market_share="Early but growing rapidly"
        ),
        news=News(
            recent_important_news=[
                NewsItem(
                    title="Tavily raises $5M to compete in the AI search space",
                    url="https://techcrunch.com/tavily-raises-5m",
                    date_published=date(2024, 11, 15)
                ),
                NewsItem(
                    title="Israeli startup Tavily launches AI-powered research tool",
                    url="https://calcalist.co.il/tavily-launches",
                    date_published=date(2024, 5, 30)
                )
            ]
        ),
        final_report=FinalReport(
            background_summary="Tavily is a promising AI startup founded in 2022, focused on transforming information retrieval.",
            financial_health_summary="With $6M in funding and a manageable burn rate, the company has a runway of about a year.",
            market_position_summary="They compete with early-stage AI tools but differentiate on orchestration and precision.",
            news_summary="Tavily recently raised Series A funding and launched a new research product."
        )
    )