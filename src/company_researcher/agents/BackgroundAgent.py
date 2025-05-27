from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
import logging
from langchain_core.messages import HumanMessage
from company_researcher.agents.BaseAgent import BaseAgent
from company_researcher.api_clients.tavily_client import PageContent, TavilyBatchSearchInput, TavilyClient
from company_researcher.workflow.langgraph_workflow import ResearchState

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
        site_content = await self.tavily_client.crawl(state['company_url'], max_depth=1, limit=5, instructions="Extract company background information.")
        logging.info(f"Extracted site content: {site_content}")
        
        
        logging.info("Summarizing site content to grounded information.")
        grounded_info = self._summarize_to_grounded_info(site_content)
        
        max_queries = self.config.get('max_queries', 3)
        search_input = self._generate_queries_for_missing_info(grounded_info, max_queries)
        search_output_for_missing_info = await self.tavily_client.search(search_input)
        
        
        # Process and summarize the background information
        background = self.process_background_data(grounded_info, search_output_for_missing_info)
        
        return {
            'background': {
                'content': background,
            }
        }
        
    def process_background_data(self, grounded_info: str, search_output: Dict[str, Any]) -> str:
        """Process the grounded information and search output to create a comprehensive background report."""
        
        prompt = f"""
        You are an expert in summarizing company backgrounds.
        Your given a grounded information about a company, and search results for missing information.
        The search results may be incomplete or contain irrelevant information. Your'e also given a score for each search result, this score (0-1) gives you an idea of how relevant the search result is to the grounded information. However, the score is not always accurate, so you should use your judgement to determine the relevance of the search results.
        Your task is to create a comprehensive background report, it must include all the information from the grounded information, and any relevant information from the search results.
        Grounded Information: {grounded_info}
        Search Results: {search_output}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
        
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
        You are an expert in summarizing company backgrounds.
        Your summary should be grounded in the provided content. Do not make assumptions or add information not present in the content.    
        Please provide a concise but comprehensive summary focusing on:
        - Company founding and history
        - Main business activities
        - Key milestones
        - Current status
        
        {content}
        
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _generate_queries_for_missing_info(self, grounded_info: str, max_queries: int) -> TavilyBatchSearchInput:
        """Generate queries for any missing information based on the grounded information."""
        
        prompt = f"""
        You are an expert in generating search queries for missing information.
        Based on the following grounded information, generate search queries to find any missing details about the company's background.
        Do not repeat information already present in the grounded information.
        Do not generate queries that are specific to finance, news or market data. You should only generate queries that are relevant to the company's background, such as its history, founding, main business activities, key milestones, and current status.
        The queries should be concise and specific, and should not exceed 10 words each.
        You should generate 0-{max_queries} queries.
        
        {grounded_info}
        """
        
        response = self.llm.with_structured_output(TavilyBatchSearchInput).invoke([HumanMessage(content=prompt)])
        return response
