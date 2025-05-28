from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
import logging
from langchain_core.messages import HumanMessage
from company_researcher.agents.BaseAgent import BaseAgent
from company_researcher.api_clients.tavily_client import PageContent, SearchResponse, TavilyBatchSearchInput, TavilyClient
from company_researcher.workflow.langgraph_workflow import ResearchState
from company_researcher.workflow.states import FinancialHealth

class FinancialHealthAgent(BaseAgent):
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
        super().__init__(llm, tavily_client, config)
        
    async def run(self, state: ResearchState) -> ResearchState:
        """
        The main method to run the financial health agent.
        """
        # Use Tavily API to search for company financial information    
        site_content = state['site_content']
        logging.info("Summarizing site content to grounded financial information.")
        grounded_financial_info = self._summarize_to_grounded_financial_info(site_content)
        
        max_queries = self.config.get('max_queries', 8)
        search_input = self._generate_queries_for_missing_financial_info(grounded_financial_info, max_queries)
        search_output_for_missing_info = await self.tavily_client.search(search_input)
        
        # Process and analyze the financial health information
        financial_analysis = self.process_financial_data(
            grounded_financial_info, 
            search_output_for_missing_info,
        )
        
        return state.model_copy(update={
            "financial_health": financial_analysis,
        })
        
    def process_financial_data(self, grounded_financial_health: FinancialHealth, search_output: List[SearchResponse]) -> FinancialHealth:
        """Process the grounded information and search output to create a comprehensive Financial Health report."""
        
        prompt = f"""
        You are an expert generating reliable financial health information about a company.
        Given:
        - A partial FinancialHealth (with some fields already populated)
        - A list of SearchResponse results. Note that some of the search results may be irrelevant or not useful. You should use your judgment to determine which results are relevant.

        For any field that is non-empty in the partial input, preserve its value exactly; fill only empty or missing fields using the search results.
        Do not invent information.

        Partial Financial Health:
        {grounded_financial_health.model_dump_json(exclude_unset=True)}

        Search results:
        {search_output}
        """
        
        return self.llm.with_structured_output(FinancialHealth).invoke([HumanMessage(content=prompt)])
        
    def _summarize_to_grounded_financial_info(self, site_content: List[PageContent]) -> str:
        """Summarize the crawled site content to grounded financial information."""
        
        if not site_content:
            logging.error("No financial content found on the site.")
            raise ValueError("No financial content found on the site.")
        
        # Use LLM to summarize the crawled financial content
        content_parts = []
        for item in site_content:
            url = item.url.strip()
            text = item.raw_content.strip()
            content_parts.append(f"URL: {url}\nContent:\n{text}")
        
        content = '\n\n'.join(content_parts)
        logging.info(f"Summarizing financial site content, number of parts: {len(content_parts)}")
        
        prompt = f"""
        You are an expert in summarizing company financial information based on their site contents.
        Your summary should be grounded in the provided content.
        Only fill fields that the content provides information for.
        If the content does not provide information for a field, leave it empty.
        You better not fill fields that the content does not provide information for than to fill them with incorrect information.
        
        {content}
        """
        
        response = self.llm.with_structured_output(FinancialHealth).invoke([HumanMessage(content=prompt)])
        return response

    def _generate_queries_for_missing_financial_info(self, grounded_info: FinancialHealth, max_queries: int) -> TavilyBatchSearchInput:
        """Generate queries for any missing financial information based on the grounded information."""
        
        prompt = f"""
        You are an expert in generating search queries for missing information.
        You are given a financial health information about a company, where some fields are missing.
        Your task is to generate search queries that their answers will fill the missing fields in the grounded information.
        You should generate 0-{max_queries} queries.
        
        {grounded_info}
        """
        
        response = self.llm.with_structured_output(TavilyBatchSearchInput).invoke([HumanMessage(content=prompt)])
        return response
