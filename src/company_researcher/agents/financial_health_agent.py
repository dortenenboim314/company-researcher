from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
import logging

from company_researcher.agents.base_agent import BaseAgent
from company_researcher.api_clients.tavily_client import PageContent, TavilyClient
from company_researcher.workflow.langgraph_workflow import ResearchState
from company_researcher.workflow.states import FinancialHealth

class FinancialHealthAgent(BaseAgent[FinancialHealth]):
    """
    Agent that gathers and analyzes financial health information for a company.
    Inherits core behavior from BaseAgent.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: TavilyClient,
        config: Dict[str, Any]
    ):
        super().__init__(FinancialHealth, llm, tavily_client, config)
        
    async def run(self, state: ResearchState) -> ResearchState:
        """
        The main method to run the financial health agent.
        Uses the generic workflow from BaseAgent.
        """
        return await self.run_agent_workflow(state)

    def get_state_field_name(self) -> str:
        """
        Returns the field name for financial health information in the state.
        """
        return "financial_health"

    def get_info_type_description(self) -> str:
        """
        Returns description for use in prompts.
        """
        return "financial health"
