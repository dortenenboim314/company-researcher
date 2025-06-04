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
            "extract_from_site_content": load_prompt("background\extract_from_site_content.txt"),
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
        generate_search_queries_prompt = f"""
        You are an expert in researching company background information. Your task is to search for and answer questions about the company {state['company_name']}. 
You conducted a web crawl and gathered some background information, but you need to fill in any gaps or missing details, as the reviewer has identified some incomplete information.
Your task is to return a list of search queries that will help us gather the missing information.
Search queries should be specific and focused on the missing or incomplete details identified by the reviewer.
Each query should be precise. Sometimes it might be useful to break down complex questions into simpler, more focused search queries.
"""
        
        search_queries = await self.llm.with_structured_output(TavilyBatchSearchInput).ainvoke([SystemMessage(content=generate_search_queries_prompt)] + state["messages"])
        search_response = await self.tavily_client.search(search_queries)
        response_str = "\n########\n".join([res.to_string() for res in search_response])
        
        answer_based_on_search_prompt = f"""
You are an expert in researching company background information. Your task is to answer the Reviewer's questions based on the search queries' results provided.
Your answer should be based only on the search results provided below. Do not add any additional information.

        Search results:
        {response_str}
        """

        response = await self.llm.ainvoke([SystemMessage(content=answer_based_on_search_prompt)] + state["messages"])
        response.name = "Researcher"
        return {"messages": [response]}

    async def _review(self, state: BackgroundResearchState) -> BackgroundResearchState:
        prompt = f"""
        You are an expert in reviewing company background information extracted by a Researcher. Your task is to review the gathered background information about the company {state['company_name']} and identify any missing or incomplete details.
Background information includes, but is not limited to:
Industry, founding date, mission or vision, notable milestones, current status, and estimated number of employees.
You should not focus on financial health, market position, or news articles.
Please provide a list of missing or incomplete details that need further research.
        """
        response = await self.llm.ainvoke([SystemMessage(content=prompt)] + state["messages"])
        response.name = "Reviewer"
        return {"messages": [response]}

    async def _summarize(self, state: BackgroundResearchState) -> BackgroundResearchState:
        prompt = f"""
        You are an expert in summarizing company background information based on a conversation between a Researcher and an Reviewer.
Your task is to create a concise summary of the gathered background information about the company {state['company_name']}.
The summary should include key details such as (but not limited to) industry, founding date, mission or vision, notable milestones, current status, and estimated number of employees.
The summary should be based on the conversation history provided below. DO NOT include any additional information or assumptions.
        """
        response = await self.llm.ainvoke([SystemMessage(content=prompt)] + state["messages"])
        response.name = "Background Information Summarizer"
        
        return {
            "company_background": response.content,
            "internal_messages": state["messages"] + [response]
        }