import operator
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from company_researcher.core.agents.background import BackgroundAgent
from company_researcher.core.agents.topic_research_agent import TopicResearchAgent
from company_researcher.core.api_clients.tavily_client import TavilyBatchSearchInput, TavilyClient
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState

class CompanyResearchInput(TypedDict):
    company_name: str
    company_url: str

class CompanyResearchOutput(TypedDict):
    final_report: str

class CompanyResearchState(MessagesState):
    company_name: str
    company_url: str
    company_background: str
    results: Annotated[list, operator.add]


class CompanyResearchAgent:
    def __init__(self,
                 llm:ChatOpenAI,
                 tavily_client:TavilyClient):
        
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
            max_steps=3
        )
        self.market_position_agent = TopicResearchAgent(
            llm=self.llm,
            tavily_client=self.tavily_client,
            topic_name="Market Position",
            topic_description="Gather and analyze the company's market position, including its competitors, market share, and industry trends.",
            max_steps=3
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
        
        self.compiled_graph = self.graph.compile()
    
    def perform_research(self, company_name: str, company_url: str) -> CompanyResearchState:
        """Perform company research by invoking the state graph."""
        research_input = CompanyResearchInput(
            company_name=company_name,
            company_url=company_url,
        )
        result = self.compiled_graph.ainvoke(research_input)
        return CompanyResearchState(**result)

    async def _summarize_results(self, state: CompanyResearchState) -> CompanyResearchState:
        prompt = f"""
        You are an expert in summarizing company research. Your task is to create a final report based on the results of the research conducted on a company.
        The research has been conducted by 3 specialized agents: Background Research, Financial Health Research, and Market Position Research.
        You should not make up any information, and you should not use any external knowledge or assumptions.
        You need to write a report to a reader who is not familiar with the company or the research process.
        The report should be concise, clear, and informative, summarizing the key findings from each research area.
        The report should be structured and easy to read. You should begin with an introduction to the company, followed by sections for each research area, and conclude with a summary of the overall findings.
        The company being researched is {state['company_name']}.
        Below are the results from each research area.
"""

        messages = [SystemMessage(content=prompt)] + state["results"]
        response = await self.llm.ainvoke(messages)
        state["final_report"] = response.content
        return state