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
            
    async def crawl(self, url: str) -> list[tuple[str, str]]:
        """
        Perform a web crawl using Tavily API.
        
        Args:
            url: The URL to crawl.
            
        Returns:
            List of (url, content) tuples from the crawl results.
        """
        logging.info(f"Starting crawl for URL: {url}")
        
        res = await self.async_client.crawl(url=url, max_depth=2, limit=20)
        
        logging.info(f"Crawl completed for URL: {url}")
        logging.debug(f"Crawl result: {res}")
        
        return res['results']
    
    
        
    async def search(self, batch_search_input: TavilyBatchSearchInput) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query: The search query string.
            
        Returns:
            Dict containing the search results from Tavily API.
        """
        
        return await asyncio.gather(
            *[self.async_client.search(query=query) for query in batch_search_input.queries]
        )
