from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
import logging

from company_researcher.agents.base_agent import BaseAgent
from company_researcher.api_clients.tavily_client import PageContent, TavilyClient
from company_researcher.workflow.langgraph_workflow import ResearchState
from company_researcher.workflow.states import CompanyBackground

class BackgroundAgent(BaseAgent[CompanyBackground]):
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
        super().__init__(CompanyBackground, llm, tavily_client, config)
        
    async def run(self, state: ResearchState) -> ResearchState:
        """
        The main method to run the background agent.
        Uses the generic workflow from BaseAgent.
        """
        logging.info(f"Running BackgroundAgent with state: {state}")
        return await self.run_agent_workflow(state)

    async def run_agent_workflow(self, state: ResearchState) -> ResearchState:
        """
        BackgroundAgent's workflow that includes adding site_content to state.
        
        Args:
            state: The current research state.
            
        Returns:
            The updated state after agent execution.
        """
        # Get site content (different for each agent)
        site_content = await self.get_site_content(state)
        
        # Summarize site content to grounded information
        logging.info("Summarizing site content to grounded information.")
        grounded_info = self._summarize_to_grounded_info(state.company_name, site_content)
        
        # Generate queries for missing information
        if self._check_missing_fields(grounded_info):
            logging.info("Missing fields detected in grounded information. Generating queries for missing data.")
            # Generate queries for missing information
            max_queries = self.config.get('max_queries', 5)
            search_input = self._generate_queries_for_missing_info(state.company_name, grounded_info, max_queries)
            
            # Search for missing information
            search_output_for_missing_info = await self.tavily_client.search(search_input)
            
            # Process and analyze the information
            final_info = self._process_data(grounded_info, search_output_for_missing_info)
        else:
            logging.info("No missing fields detected. Using grounded information directly.")
            final_info = grounded_info
        
        # Prepare state updates including site_content for other agents
        state_updates = {
            self.get_state_field_name(): final_info,
            "current_step": self.__class__.__name__,
            'site_content': site_content  # BackgroundAgent's special behavior
        }
        
        return state.model_copy(update=state_updates)

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
