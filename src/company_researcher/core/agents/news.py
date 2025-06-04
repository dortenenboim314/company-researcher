from typing import List, Optional, TypedDict
from langchain_openai import ChatOpenAI
from company_researcher.core.api_clients.tavily_client import TavilyClient
from langgraph.graph import StateGraph, END, START
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

class NewsInput(TypedDict):
    company_name: str
    company_background: str


class NewsItem(BaseModel):
    title: str = Field(..., description="Headline of the news item")
    url: str = Field(..., description="Link to the news article")
    date_published: Optional[str] = Field(None, description="Publication date in ISO format YYYY-MM-DD", example="2023-10-17")
    
class NewsOutput(BaseModel):
    recent_important_news: List[NewsItem] = Field(
        description="List of recent important news items"
    )

class NewsResearchState(MessagesState):
    company_name: str
    company_url: str
    result: NewsOutput


class NewsAgent:
    def __init__(self,
                 llm:ChatOpenAI,
                 tavily_client:TavilyClient):
        
        self.llm = llm
        self.tavily_client = tavily_client
        
        self.graph = StateGraph(state_schema=NewsResearchState,
                                input=NewsInput,
                                output=NewsOutput)

        self.graph.add_node("gather_news", self.gather_news)
        
        # start -> gather_news -> end
        self.graph.add_edge(START, "gather_news")
        self.graph.add_edge("gather_news", END)
        
    async def gather_news(self, state: NewsResearchState) -> NewsResearchState:
        self.tavily_client.search(batch_search_input=search_input)
     