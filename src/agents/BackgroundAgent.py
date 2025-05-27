import os
import sys
from typing import Any, Dict, List, TypedDict
from langchain_openai import ChatOpenAI
import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.api_clients.tavily_client import TavilyBatchSearchInput, TavilyClient

# Define ResearchState locally to avoid import issues
class ResearchState(TypedDict):
    company_name: str
    company_url: str
    company_background: str
    financial_data: Dict[str, Any]
    market_position: str
    recent_news: List[Dict[str, str]]
    final_report: str
    current_step: str
    errors: List[str]

class BackgroundAgent:
    """
    A class representing a background agent that performs tasks in the background.
    This class is designed to be extended by other agents that require background processing.
    """

    def __init__(self, llm: ChatOpenAI, tavily_client: TavilyClient, config):
        """
        Initializes the BackgroundAgent.
        """
        self.llm = llm
        self.tavily_client = tavily_client
        self.config = config
        
    async def run(self, state: ResearchState) -> ResearchState:
        """
        The main method to run the background agent.
        This method should be overridden by subclasses to implement specific background tasks.
        """
        # Use Tavily API to search for company background
        logging.info(f"Running BackgroundAgent with state: {state}")
        site_content = await self.tavily_client.crawl(state['company_url'])
        logging.debug(f"Extracted site content: {site_content}")
        
        
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
        
    def _summarize_to_grounded_info(self, site_content: List[Dict[str, Any]]) -> str:
        """Summarize the crawled site content to grounded information."""
        if not site_content:
            return "No content found on the site."
        
        # Use LLM to summarize the crawled content
        # Handle the actual format returned by Tavily crawl
        content_parts = []
        for item in site_content:
            url = item.get('url', 'Unknown URL')
            text = item.get('content', '')
            if text:
                content_parts.append(f"URL: {url}\nContent: {text}")
        
        content = '\n\n'.join(content_parts)
        
        if not content.strip():
            return "No meaningful content found on the site."
        
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

    def _generate_queries_for_missing_info(self, grounded_info: str, max_queries: int = 3) -> TavilyBatchSearchInput:
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

# write a main function to test the BackgroundAgent class
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    from src.api_clients.tavily_client import TavilyClient
    from langchain_openai import ChatOpenAI

    load_dotenv()

    async def main():
        llm = ChatOpenAI(model="gpt-4")
        tavily_client = TavilyClient(api_key="your_tavily_api_key")
        agent = BackgroundAgent(llm, tavily_client, config={})

        state = ResearchState(company_url="https://dreamgroup.com")
        result = await agent.run(state)
        print(result)

    asyncio.run(main())
