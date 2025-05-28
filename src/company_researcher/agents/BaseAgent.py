from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import logging
from pydantic import BaseModel

from company_researcher.api_clients.tavily_client import PageContent, SearchResponse, TavilyBatchSearchInput, TavilyClient
from company_researcher.workflow.langgraph_workflow import ResearchState

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the company_researcher project.

    All agents must accept an LLM, a TavilyClient, and a configuration dict,
    and implement an asynchronous `run` method to perform their task.
    
    This class provides common workflow patterns shared by multiple agents.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: TavilyClient,
        config: Dict[str, Any]
    ):
        """
        Initializes the BaseAgent with core dependencies.

        Args:
            llm: An instance of ChatOpenAI for language model calls.
            tavily_client: An instance of TavilyClient for data retrieval.
            config: A configuration dictionary with agent-specific settings.
        """
        self.llm = llm
        self.tavily_client = tavily_client
        self.config = config

    @abstractmethod
    async def run(self, state: ResearchState) -> ResearchState:
        """
        Main entry point for the agent. Must be implemented by subclasses.

        Args:
            state: The current research state.

        Returns:
            The updated state after agent execution.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    @abstractmethod
    async def get_site_content(self, state: ResearchState) -> List[PageContent]:
        """
        Get site content - either by crawling or from existing state.
        
        Args:
            state: The current research state.
            
        Returns:
            List of PageContent objects.
        """
        raise NotImplementedError("Subclasses must implement get_site_content method.")

    @abstractmethod
    def get_output_type(self) -> Type[BaseModel]:
        """
        Get the structured output type for this agent.
        
        Returns:
            The Pydantic model class for structured output.
        """
        raise NotImplementedError("Subclasses must implement get_output_type method.")

    @abstractmethod
    def get_state_field_name(self) -> str:
        """
        Get the field name in the state that this agent updates.
        
        Returns:
            The field name as a string.
        """
        raise NotImplementedError("Subclasses must implement get_state_field_name method.")

    @abstractmethod
    def get_info_type_description(self) -> str:
        """
        Get a description of the type of information this agent processes.
        
        Returns:
            Description string for use in prompts.
        """
        raise NotImplementedError("Subclasses must implement get_info_type_description method.")

    def get_additional_state_updates(self, state: ResearchState, site_content: List[PageContent]) -> Dict[str, Any]:
        """
        Get any additional state updates specific to the agent.
        
        Args:
            state: The current research state.
            site_content: The site content retrieved.
            
        Returns:
            Dictionary of additional state updates.
        """
        return {}

    async def run_agent_workflow(self, state: ResearchState) -> ResearchState:
        """
        Generic workflow that can be used by most agents.
        
        Args:
            state: The current research state.
            
        Returns:
            The updated state after agent execution.
        """
        # Get site content (different for each agent)
        site_content = await self.get_site_content(state)
        
        # Summarize site content to grounded information
        logging.info("Summarizing site content to grounded information.")
        grounded_info = self._summarize_to_grounded_info(site_content)
        
        # Generate queries for missing information
        max_queries = self.config.get('max_queries', 5)
        search_input = self._generate_queries_for_missing_info(grounded_info, max_queries)
        
        # Search for missing information
        search_output_for_missing_info = await self.tavily_client.search(search_input)
        
        # Process and analyze the information
        final_info = self._process_data(grounded_info, search_output_for_missing_info)
        
        # Prepare state updates
        state_updates = {
            self.get_state_field_name(): final_info,
            "current_step": self.__class__.__name__
        }
        
        # Add any additional state updates
        additional_updates = self.get_additional_state_updates(state, site_content)
        state_updates.update(additional_updates)
        
        return state.model_copy(update=state_updates)

    def _summarize_to_grounded_info(self, site_content: List[PageContent]) -> BaseModel:
        """Summarize the crawled site content to grounded information."""
        
        if not site_content:
            logging.error(f"No {self.get_info_type_description()} content found on the site.")
            raise ValueError(f"No {self.get_info_type_description()} content found on the site.")
        
        # Use LLM to summarize the crawled content
        content_parts = []
        for item in site_content:
            url = item.url.strip()
            text = item.raw_content.strip()
            content_parts.append(f"URL: {url}\nContent:\n{text}")
        
        content = '\n\n'.join(content_parts)
        logging.info(f"Summarizing {self.get_info_type_description()} site content, number of parts: {len(content_parts)}")
        
        prompt = f"""
        You are an expert in summarizing company {self.get_info_type_description()} based on their site contents.
        Your summary should be grounded in the provided content.
        Only fill fields that the content provides information for.
        If the content does not provide information for a field, leave it empty.
        You better not fill fields that the content does not provide information for than to fill them with incorrect information.
        
        {content}
        """
        
        response = self.llm.with_structured_output(self.get_output_type()).invoke([HumanMessage(content=prompt)])
        return response

    def _generate_queries_for_missing_info(self, grounded_info: BaseModel, max_queries: int) -> TavilyBatchSearchInput:
        """Generate queries for any missing information based on the grounded information."""
        
        prompt = f"""
        You are an expert in generating search queries for missing information.
        You are given a {self.get_info_type_description()} information about a company, where some fields are missing.
        Your task is to generate search queries that their answers will fill the missing fields in the grounded information.
        You should generate 0-{max_queries} queries.
        
        {grounded_info}
        """
        
        response = self.llm.with_structured_output(TavilyBatchSearchInput).invoke([HumanMessage(content=prompt)])
        return response

    def _process_data(self, grounded_info: BaseModel, search_output: List[SearchResponse]) -> BaseModel:
        """Process the grounded information and search output to create a comprehensive report."""
        
        prompt = f"""
        You are an expert generating reliable {self.get_info_type_description()} information about a company.
        Given:
        - A partial {self.get_output_type().__name__} (with some fields already populated)
        - A list of SearchResponse results. Note that some of the search results may be irrelevant or not useful. You should use your judgment to determine which results are relevant.

        For any field that is non-empty in the partial input, preserve its value exactly; fill only empty or missing fields using the search results.
        Do not invent information.

        Partial {self.get_info_type_description()}:
        {grounded_info.model_dump_json(exclude_unset=True)}

        Search results:
        {search_output}
        """
        
        return self.llm.with_structured_output(self.get_output_type()).invoke([HumanMessage(content=prompt)])
