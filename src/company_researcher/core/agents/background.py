from typing import TypedDict
from langchain_openai import ChatOpenAI
from company_researcher.core.api_clients.tavily_client import TavilyBatchSearchInput, TavilyClient
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from company_researcher.core.agents.prompts.utils import load_prompt
import logging

class BackgroundInput(TypedDict):
    company_name: str
    company_url: str

class BackgroundOutput(TypedDict):
    company_background: str
    internal_messages: list
    
class BackgroundResearchState(MessagesState):
    company_name: str
    company_url: str
    company_background: str


class BackgroundAgent:
    def __init__(self,
                 llm:ChatOpenAI,
                 tavily_client:TavilyClient):
        
        self.llm = llm
        self.tavily_client = tavily_client
        
        self.graph = StateGraph(state_schema=BackgroundResearchState,
                                input=BackgroundInput,
                                output=BackgroundOutput)
        
        self.graph.add_node("crawl_and_gather_background", self._crawl_and_gather_background)
        self.graph.add_node("search_and_answer", self._search_and_answer)
        self.graph.add_node("review", self._review)
        self.graph.add_node("summarize", self._summarize)

        # start -> crawl_and_gather_background -> review -> search_and_answer -> summarize -> end
        self.graph.add_edge(START, "crawl_and_gather_background")
        self.graph.add_edge("crawl_and_gather_background", "review")
        self.graph.add_edge("review", "search_and_answer")
        self.graph.add_edge("search_and_answer", "summarize")
        self.graph.add_edge("summarize", END)
        
        self.prompts = {
            "extract_from_site_content": load_prompt("background/extract_from_site_content.txt"),
            "generate_search_queries": load_prompt("background/generate_search_queries.txt"),
            "answer_based_on_search": load_prompt("background/answer_based_on_search.txt"),
            "review": load_prompt("background/review.txt"),
            "summarize": load_prompt("background/summarize.txt"),
        }
    
    def compile(self) -> StateGraph:
        """Compile the state graph for the agent.

        Returns:
            StateGraph: The compiled state graph.
        """
        return self.graph.compile()
    
    async def _crawl_and_gather_background(self, state: BackgroundResearchState) -> BackgroundResearchState:
        site_contents = await self.tavily_client.crawl(state["company_url"], max_depth=2, limit=5, instructions=f"Gather background information about the company {state['company_name']}.")
        site_contents_str = "\n######\n".join([site.to_string() for site in site_contents])
        
        prompt = self.prompts["extract_from_site_content"].format(site_contents_str=site_contents_str, company_name=state["company_name"])
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        response.name = "Researcher"
        return {"messages": [response]} 

    async def _search_and_answer(self, state: BackgroundResearchState) -> BackgroundResearchState:
        generate_search_queries_prompt = self.prompts["generate_search_queries"].format(company_name=state["company_name"])
        
        search_queries = await self.llm.with_structured_output(TavilyBatchSearchInput).ainvoke([SystemMessage(content=generate_search_queries_prompt)] + state["messages"])
        search_response = await self.tavily_client.search(search_queries)
        response_str = "\n########\n".join([res.to_string() for res in search_response])
        
        answer_based_on_search_prompt = self.prompts["answer_based_on_search"].format(response_str=response_str)

        response = await self.llm.ainvoke([SystemMessage(content=answer_based_on_search_prompt)] + state["messages"])
        response.name = "Researcher"
        return {"messages": [response]}

    async def _review(self, state: BackgroundResearchState) -> BackgroundResearchState:
        prompt = self.prompts["review"].format(company_name=state["company_name"])
        response = await self.llm.ainvoke([SystemMessage(content=prompt)] + state["messages"])
        response.name = "Reviewer"
        return {"messages": [response]}

    async def _summarize(self, state: BackgroundResearchState) -> BackgroundResearchState:
        prompt = self.prompts["summarize"].format(company_name=state["company_name"])
        response = await self.llm.ainvoke([SystemMessage(content=prompt)] + state["messages"])
        response.name = "Background Information Summarizer"
        
        return {
            "company_background": response.content,
            "internal_messages": state["messages"] + [response]
        }
