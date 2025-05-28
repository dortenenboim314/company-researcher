from typing import Any, Dict, List, Type
from langchain_openai import ChatOpenAI
import logging
from pydantic import BaseModel

from company_researcher.agents.BaseAgent import BaseAgent
from company_researcher.api_clients.tavily_client import PageContent, TavilyClient
from company_researcher.workflow.langgraph_workflow import ResearchState
from company_researcher.workflow.states import CompanyBackground

class BackgroundAgent(BaseAgent):
    """
    Agent that gathers and summarizes background information for a company.
    Inherits core behavior from BaseAgent.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: TavilyClient,
        config: Dict[str, Any]
    ):
        super().__init__(llm, tavily_client, config)
        
    async def run(self, state: ResearchState) -> ResearchState:
        """
        The main method to run the background agent.
        Uses the generic workflow from BaseAgent.
        """
        logging.info(f"Running BackgroundAgent with state: {state}")
        return await self.run_agent_workflow(state)

    async def get_site_content(self, state: ResearchState) -> List[PageContent]:
        """
        BackgroundAgent crawls the site content.
        """
        site_content = await self.tavily_client.crawl(
            state.company_url, 
            max_depth=2, 
            limit=10, 
            instructions=f"Extract company information about {state.company_name}."
        )
        logging.info(f"Extracted site content: {site_content}")
        return site_content

    def get_output_type(self) -> Type[BaseModel]:
        """
        Returns the CompanyBackground model for structured output.
        """
        return CompanyBackground

    def get_state_field_name(self) -> str:
        """
        Returns the field name for background information in the state.
        """
        return "background"

    def get_info_type_description(self) -> str:
        """
        Returns description for use in prompts.
        """
        return "background"

    def get_additional_state_updates(self, state: ResearchState, site_content: List[PageContent]) -> Dict[str, Any]:
        """
        BackgroundAgent adds site_content to the state for other agents to use.
        """
        return {
            'site_content': site_content
        }
