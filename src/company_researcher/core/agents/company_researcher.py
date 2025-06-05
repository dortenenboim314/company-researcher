import operator
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from company_researcher.config.config import Config
from company_researcher.core.agents import TopicResearchAgent, BackgroundAgent
from company_researcher.core.api_clients import TavilyClient
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import MessagesState
from company_researcher.core.agents.prompts.utils import load_prompt
from pydantic import BaseModel, Field
import logging

class CompanyResearchInput(TypedDict):
    company_name: str
    company_url: str

class GroundedInformation(BaseModel):
    background: str = Field(description="Background information such as (but not limited to) its industry, founding date, mission or vision, notable milestones, current status, and estimated number of employees.")
    financial_health: str = Field(description="Financial health information including revenue, expenses, and profitability.")
    market_position: str = Field(description="Market position information including competitors, market share, and industry trends.")

class CompanyResearchOutput(BaseModel):
    grounded_information: GroundedInformation = Field(description="Grounded information about the company, including background, financial health, and market position. should contain only information from the research conducted by the agents.")
    positive_aspects: list[str] = Field(description="List of Positive aspects of the company, such as strengths, opportunities, and positive trends.")
    negative_aspects: list[str] = Field(description="List of Negative aspects of the company, such as weaknesses, threats, and negative trends.")

class CompanyResearchState(MessagesState):
    company_name: str
    company_url: str
    company_background: str
    results: Annotated[list, operator.add]


class CompanyResearchAgent:
    def __init__(self,
                 llm:ChatOpenAI,
                 tavily_client:TavilyClient,
                 config: Config):
        
        self.llm = llm
        self.tavily_client = tavily_client
        
        self.graph = StateGraph(state_schema=CompanyResearchState,
                                input=CompanyResearchInput,
                                output=CompanyResearchOutput)
        self.background_agent = BackgroundAgent(llm=self.llm, tavily_client=self.tavily_client)
        self.financial_health_agent = TopicResearchAgent(
            llm=self.llm,
            tavily_client=self.tavily_client,
            topic_name="Financial Health",
            topic_description="Gather and analyze financial health information for the company, including revenue, expenses, and profitability.",
            max_steps=config.max_searches_per_agent
        )
        self.market_position_agent = TopicResearchAgent(
            llm=self.llm,
            tavily_client=self.tavily_client,
            topic_name="Market Position",
            topic_description="Gather and analyze the company's market position, including its competitors, market share, and industry trends.",
            max_steps=config.max_searches_per_agent
        )


        self.graph.add_node("background_research", self.background_agent.compile())
        self.graph.add_node("financial_health_research", self.financial_health_agent.compile())
        self.graph.add_node("market_position_research", self.market_position_agent.compile())
        self.graph.add_node("summarize_results", self._summarize_results)
        
        # Start -> background_research -> (financial_health_research, market_position_research) -> summarize_results -> End
        self.graph.add_edge(START, "background_research")
        self.graph.add_edge("background_research", "financial_health_research")
        self.graph.add_edge("background_research", "market_position_research")
        self.graph.add_edge("financial_health_research", "summarize_results")
        self.graph.add_edge("market_position_research", "summarize_results")
        self.graph.add_edge("summarize_results", END)
        
        self.prompts = {
            "summarize_results": load_prompt("company_researcher/summarize_results.txt"),
        }
        
        self.compiled_graph = self.graph.compile()
    
    async def perform_research(self, company_name: str, company_url: str) -> CompanyResearchOutput:
        """Perform company research by invoking the state graph."""
        research_input = CompanyResearchInput(
            company_name=company_name,
            company_url=company_url,
        )
        result = await self.compiled_graph.ainvoke(research_input)
        research_output = CompanyResearchOutput(**result)
        return research_output

    async def _summarize_results(self, state: CompanyResearchState) -> CompanyResearchState:
        background_report = f"Background Research:\n{state['company_background']}\n"
        reports = [background_report] + [msg.content for msg in state['results']]
        reports = "\n####\n".join(reports)
        prompt = self.prompts["summarize_results"].format(company_name=state["company_name"], reports=reports)
        
        logging.info(f"prompt for summarization:\n{prompt}")
        
        messages = [SystemMessage(content=prompt)]
        response = await self.llm.with_structured_output(CompanyResearchOutput).ainvoke(messages)
        return response
