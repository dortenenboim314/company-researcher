"""
Tavily API client for web search functionality.
"""

import asyncio
import os
from pydantic import BaseModel, Field
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
import logging


# Load environment variables
load_dotenv()

class TavilyBatchSearchInput(BaseModel):
    queries: List[str] = Field(description="List of search queries to perform.")

class PageContent(BaseModel):
    url: str = Field(description="The URL to crawl.")
    raw_content: str = Field(description="The content of the page to be crawled.")
    
    class Config:
        allow_population_by_field_name = True
        
class ResultCandidate(BaseModel):
    title: str = Field(description="The title of the search result.")
    url: str = Field(description="The URL of the search result.")
    content: str = Field(description="A short description of the search result.")
    score: float = Field(description="Relevance score of the search result.")
    
    class Config:
        allow_population_by_field_name = True
        
class SearchResponse(BaseModel):
    query: str = Field(description="The search query used.")
    answer: Optional[str] = Field(default=None, description="The answer to the search query.")
    candidates: List[ResultCandidate] = Field(alias="results",description="List of search results.")
    
    class Config:
        allow_population_by_field_name = True

class TavilyClient:
    """
    A simple client for interacting with the Tavily search API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Tavily client.
        
        Args:
            api_key: Tavily API key. If None, will try to load from TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.async_client = AsyncTavilyClient(api_key=self.api_key)
        
        if not self.api_key:
            print("⚠️ TAVILY_API_KEY not found in environment variables")
            
    async def crawl(self, url: str, max_depth, limit, instructions=None) -> list[PageContent]:
        """
        Perform a web crawl using Tavily API.
        
        Args:
            url: The URL to crawl.
            
        Returns:
            List of dicts containing the crawled data.
        """
        logging.info(f"Starting crawl for URL: {url}")
        
        res = await self.async_client.crawl(url=url, max_depth=max_depth, limit=limit, instructions=instructions)
        
        logging.info(f"Crawl completed for URL: {url}")
        logging.debug(f"Crawl result: {res}")
        
        pages = [PageContent(**d) for d in res['results']]

        logging.info(f"Extracted {len(pages)} pages from crawl.")
        return pages
        
    
    
        
    async def search(self, batch_search_input: TavilyBatchSearchInput) -> list[SearchResponse]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query: The search query string.
            
        Returns:
            Dict containing the search results from Tavily API.
        """
        
        results = await asyncio.gather(
            *[self.async_client.search(query=query) for query in batch_search_input.queries]
        )
        
        # Debug: Log the actual response structure
        if results:
            logging.info(f"Sample API response structure: {list(results[0].keys())}")
            logging.info(f"Sample API response: {results[0]}")
        
        return [SearchResponse(**res) for res in results if res]
