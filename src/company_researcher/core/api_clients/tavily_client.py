"""
Tavily API client for web search functionality.
"""

import asyncio
import os
from pydantic import BaseModel, Field
from typing import List, Optional
from tavily import AsyncTavilyClient
import logging
import re


class TavilyBatchSearchInput(BaseModel):
    queries: List[str] = Field(description="List of search queries to perform.")

class PageContent(BaseModel):
    url: str = Field(description="The URL to crawl.")
    raw_content: str = Field(description="The content of the page to be crawled.")
    
    def to_string(self) -> str:
        return f"URL: {self.url}\nRaw Content: {self.raw_content}"
    
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
    
    def to_string(self, top_k_candidates: int = 3) -> str:
        info = []
        if self.query:
            info.append(f"Query: {self.query}")
        if self.answer:
            info.append(f"Snippet: {self.answer}")
        if self.candidates:
            info.append("Results:")
            for result in sorted(self.candidates, key=lambda x: x.score, reverse=True)[:top_k_candidates]:
                info.append(f"- {result.title} ({result.url})")
                if result.content:
                    info.append(f"  Snippet: {result.content}")
                if result.score:
                        info.append(f"  Score: {result.score:.2f}")
            
        return '\n'.join(info)

    class Config:
        allow_population_by_field_name = True

class TavilyClient:
    """
    A simple client for interacting with the Tavily search API.
    """
    
    def __init__(self):
        """
        Initialize the Tavily client.
        """
        self.api_key = os.getenv("TAVILY_API_KEY")
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
        
        pages = []
        for d in res.get('results', []):
            raw = d.get('raw_content', '')
            logging.info(f"Raw content length: {len(raw)}")
            cleaned = TavilyClient._clean_raw_content(raw)
            logging.info(f"Cleaned content length: {len(cleaned)}")
            pages.append(PageContent(url=d.get('url', ''), raw_content=cleaned))

        logging.info(f"Extracted {len(pages)} pages from crawl.")
        return pages

    async def search(self, batch_search_input: TavilyBatchSearchInput, **kwargs) -> List[SearchResponse]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query: The search query string.
            
        Returns:
            Dict containing the search results from Tavily API.
        """
        
        logging.info(f"Starting search for {len(batch_search_input.queries)} queries.")
        
        results = await asyncio.gather(
            *[self.async_client.search(query=query, include_answer=True, **kwargs) for query in batch_search_input.queries]
        )
        
        logging.info(f"Search completed, got {len(results)} results.")
        
        return [SearchResponse(**res) for res in results if res]

    @staticmethod
    def _clean_raw_content(text: str) -> str:
        # 1) Remove Markdown links [text](url)
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        # 2) Remove any standalone http(s) URLs
        text = re.sub(r"https?://\S+", "", text)
        # 3) Collapse multiple blank lines into one
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()