
from langchain_openai import ChatOpenAI
from company_researcher.core.agents import BackgroundAgent, FinancialHealthAgent, MarketPositionAgent, NewsAgent
from company_researcher.core.api_clients.tavily_client import TavilyClient
from company_researcher.core.utils.llm_wrapper import LLMLoggingWrapper
from company_researcher.workflow.states import InputState, ResearchState
from langgraph.graph import StateGraph, END

class ResearchManager:
    def __init__(self, config: dict):
        self.llm = LLMLoggingWrapper(ChatOpenAI(
            model=config['llm_model'], 
            temperature=0,
        ))
        self.tavily_client = TavilyClient()
        self.background_agent = BackgroundAgent(llm=self.llm, tavily_client=self.tavily_client, config=config)
        self.financial_agent = FinancialHealthAgent(llm=self.llm, tavily_client=self.tavily_client, config=config)
        self.market_position_agent = MarketPositionAgent(llm=self.llm, tavily_client=self.tavily_client, config=config)
        self.news_agent = NewsAgent(llm=self.llm, tavily_client=self.tavily_client, config=config)
        
        self.graph = self._init_graph()
        
    async def perfrom_research(self, research_input: ResearchState) -> ResearchState:
        result = await self.graph.ainvoke(research_input)
        result = ResearchState(**result)
        return result

    def _init_graph(self):
        # Create the state graph
        workflow = StateGraph(ResearchState)
        
        # Add all agent nodes
        workflow.add_node(self.background_agent.get_agent_name(), self.background_agent.run)
        workflow.add_node(self.financial_agent.get_agent_name(), self.financial_agent.run)
        workflow.add_node(self.market_position_agent.get_agent_name(), self.market_position_agent.run)
        workflow.add_node(self.news_agent.get_agent_name(), self.news_agent.run)
        
        workflow.set_entry_point(self.background_agent.get_agent_name())
        workflow.add_edge(self.background_agent.get_agent_name(), self.financial_agent.get_agent_name())
        workflow.add_edge(self.financial_agent.get_agent_name(), self.market_position_agent.get_agent_name())
        workflow.add_edge(self.market_position_agent.get_agent_name(), self.news_agent.get_agent_name())
        workflow.add_edge(self.news_agent.get_agent_name(), END)
        
        # Compile the workflow
        graph = workflow.compile()
        return graph