from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
import logging
from langchain_core.messages import HumanMessage
from company_researcher.agents.BaseAgent import BaseAgent
from company_researcher.api_clients.tavily_client import PageContent, SearchResponse, TavilyBatchSearchInput, TavilyClient
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
        """
        # Use Tavily API to search for company background
        logging.info(f"Running BackgroundAgent with state: {state}")
        site_content = await self.tavily_client.crawl(state.company_url, max_depth=2, limit=10, instructions=f"Extract company information about {state.company_name}.")
        logging.info(f"Extracted site content: {site_content}")
        
        
        logging.info("Summarizing site content to grounded information.")
        grounded_company_background = self._summarize_to_grounded_info(site_content)
        
        max_queries = self.config.get('max_queries', 3)
        search_input = self._generate_queries_for_missing_info(grounded_company_background, max_queries)
        search_output_for_missing_info = await self.tavily_client.search(search_input)
        
        
        # Process and summarize the background information
        final_background = self.process_background_data(grounded_company_background, search_output_for_missing_info)
        
        return state.model_copy(update={
            'site_content': site_content,
            "background": final_background,
            "current_step": "BackgroundAgent"
        })
        
    def process_background_data(self, grounded_info: CompanyBackground, search_output: List[SearchResponse]) -> str:
        """Process the grounded information and search output to create a comprehensive background report."""
        
        prompt = f"""
        You are an expert in summarizing company backgrounds into a structured CompanyBackground JSON.
        Given:
        - A partial CompanyBackground (with some fields already populated)
        - A list of SearchResponse results.  Note that some of the search results may be irrelevant or not useful. You should use your judgment to determine which results are relevant.

        Produce a complete CompanyBackground JSON.
        For any field that is non-empty in the partial input, preserve its value exactly; fill only empty or missing fields using the search results.
        Do not invent information.

        Partial background (JSON):
        {grounded_info.model_dump_json(exclude_unset=True)}

        Search results:
        {search_output}
        """
        
        new_background = self.llm.with_structured_output(CompanyBackground).invoke([HumanMessage(content=prompt)])
        return new_background
        
    def _summarize_to_grounded_info(self, site_content: List[PageContent]) -> str:
        """Summarize the crawled site content to grounded information."""
        
        if not site_content:
            logging.error("No content found on the site.")
            raise ValueError("No content found on the site.")
        
        # Use LLM to summarize the crawled content
        # Handle the actual format returned by Tavily crawl
        content_parts = []
        for item in site_content:
            url = item.url.strip()
            text = item.raw_content.strip()
            content_parts.append(f"URL: {url}\nContent:\n{text}")
        
        content = '\n\n'.join(content_parts)
        logging.info(f"summarizing site content, number of parts: {len(content_parts)}")
        
        prompt = f"""
        You are an expert in summarizing company backgrounds based on their site contents.
        Your summary should be grounded in the provided content.
        Only fill fields that the content provides information for.
        If the content does not provide information for a field, leave it empty.
        You better not fill fields that the content does not provide information for than to fill them with incorrect information.
        
        {content}
        
        """
        response = self.llm.with_structured_output(CompanyBackground).invoke([HumanMessage(content=prompt)])
        return response

    def _generate_queries_for_missing_info(self, grounded_info: CompanyBackground, max_queries: int) -> TavilyBatchSearchInput:
        """Generate queries for any missing information based on the grounded information."""
        
        prompt = f"""
        You are an expert in generating search queries for missing information.
        You are given a grounded information about a company, where some fields are missing.
        Your task is to generate search queries that their answers will fill the missing fields in the grounded information.
        You should generate 0-{max_queries} queries.
        
        {grounded_info}
        """
        
        response = self.llm.with_structured_output(TavilyBatchSearchInput).invoke([HumanMessage(content=prompt)])
        return response
