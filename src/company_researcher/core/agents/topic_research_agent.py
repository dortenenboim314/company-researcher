from typing import TypedDict
from langchain_openai import ChatOpenAI
from company_researcher.core.api_clients.tavily_client import TavilyBatchSearchInput, TavilyClient
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
import logging

class TopicResearchInput(TypedDict):
    company_name: str
    company_background: str

class TopicResearchOutput(TypedDict):
    results: list
    internal_messages: list
    
class TopicResearchState(MessagesState):
    company_name: str
    company_background: str
    results: list
    
    
class TopicResearchAgent:
    def __init__(self,
                 llm:ChatOpenAI,
                 tavily_client:TavilyClient,
                 topic_name:str,
                 topic_description:str,
                 max_steps:int):
        """Agent for researching a specific topic related to a company.

        Args:
            llm (ChatOpenAI): the llm to use for generating responses
            topic_name (str): the name of the topic to research, e.g "Financial Health"
            topic_description (str): a description of the topic. e.g "Gather and analyze financial health information for the company, including revenue, expenses, and profitability."
        """
        
        self.llm = llm
        self.tavily_client = tavily_client
        self.topic_name = topic_name
        self.topic_description = topic_description
        self.max_steps = max_steps
        
        self.graph = StateGraph(state_schema=TopicResearchState,
                                input=TopicResearchInput,
                                output=TopicResearchOutput)
        
        self.graph.add_node("ask_question", self._ask_question)
        self.graph.add_node("search_web_and_answer", self._search_web_and_answer)
        self.graph.add_node("summarize_results", self._summarize_results)
        
        self.graph.add_edge(START, "ask_question")
        self.graph.add_conditional_edges("ask_question", self.route_to_search_or_summarize, ["search_web_and_answer", "summarize_results"])
        self.graph.add_edge("search_web_and_answer", "ask_question")
        self.graph.add_edge("summarize_results", END)
        
    def compile(self) -> StateGraph:
        """Compile the state graph for the agent.

        Returns:
            StateGraph: The compiled state graph.
        """
        return self.graph.compile()
        
    async def _summarize_results(self, state: TopicResearchState) -> TopicResearchState:
        """Summarize the results of the research.

        Args:
            state (TopicResearchState): The current state of the research.

        Returns:
            TopicResearchState: The updated state with the summary.
        """
        # This should be replaced with the actual logic to summarize the results
        prompt_for_summarizing_results = f"""You are an expert in summarizing conversations between an interviewer and an expert into a {self.topic_name} report.
        {self.topic_name} is defined as: {self.topic_description}.
        You will be given a conversation between an interviewer and an expert.
        The Interview is about {state["company_name"]}'s {self.topic_name}.
        You are also given the following background information about the company:
{state["company_background"]}
        Your task is to summarize the conversation and provide a concise report that highlights the key findings and insights related to {self.topic_name} for {state["company_name"]}.
        The report should be based only on the conversation and the background information provided, and should not include any additional information or assumptions.
        """

        summary = await self.llm.ainvoke([SystemMessage(content=prompt_for_summarizing_results)] + state["messages"])
        summary.content = f"Summary of {self.topic_name} research for {state['company_name']}:\n{summary.content}"
        logging.info(f"Summary for {self.topic_name} research:\n{summary.content}")
        return {"results": [summary]}
        
    def route_to_search_or_summarize(self, state: TopicResearchState) -> str:
        """Decide whether to search the web or summarize based on the current state.

        Args:
            state (TopicResearchState): The current state of the research.

        Returns:
            str: The next step in the workflow, either "search_web" or "summarize_results".
        """
        # count the number of messages sent by the expert
        expert_messages = [msg for msg in state["messages"] if msg.name == "Expert"]
        last_interviewer_message = state["messages"][-1].content if state["messages"] else None
        if len(expert_messages) >= self.max_steps:
            logging.info(f"Maximum steps reached for {self.topic_name} research. Summarizing results.")
            return "summarize_results"
        elif "thank" in last_interviewer_message.lower():
            logging.info(f"Interviewer asked to finish the interview.")
            return "summarize_results"
        else:
            return "search_web_and_answer"

    async def _ask_question(self, state: TopicResearchState) -> TopicResearchState:
        """Prompt an LLM to be a researcher and to ask questions regarding the topic.

        Args:
            state (TopicResearchState): _description_

        Returns:
            TopicResearchState: _description_
        """
        print(state)
        prompt_for_researcher = f"""You are an Interviewer tasked with asking an expert questions about {state["company_name"]}'s {self.topic_name}.
Where {self.topic_name} is defined as: {self.topic_description}.
You are also given the following background information about the company:
{state["company_background"]}

Your goal is to gather detailed information about {state["company_name"]}'s {self.topic_name} only by asking the expert relevant questions.
In case you think you already have enough information, you should finish the interview by saying exactly "Thank you" and nothing else."""
        messages = state["messages"]
        
        question = await self.llm.ainvoke([SystemMessage(content=prompt_for_researcher)] + messages)
        question.name = "Interviewer"
        
        return {"messages": [question]}
    
    async def _search_web_and_answer(self, state: TopicResearchState) -> TopicResearchState:
        """Search the web for information related to the topic.

        Args:
            state (TopicResearchState): _description_

        Returns:
            TopicResearchState: _description_
        """
        # This should be replaced with the actual logic to search the web
        
        prompt_asking_for_search_queries = f"""You will be given a conversation between an interviewer and an expert.
        The Interview is about {state["company_name"]}'s {self.topic_name}, which is defined as: {self.topic_description}.
        You are also given the following background information about the company:
{state["company_background"]}

Your goal is to generate a well-structured search queries.
The queries should be based on the final message of the interviewer.
Each query should be precise. Sometimes it might be useful to break down complex questions into simpler, more focused search queries."""

        search_queries = await self.llm.with_structured_output(TavilyBatchSearchInput).ainvoke([SystemMessage(content=prompt_asking_for_search_queries)] + state["messages"])
        
        # TODO - async all the way, make the graph async
        tavily_responses = await self.tavily_client.search(search_queries)

        search_results = "\n########\n".join([res.to_string() for res in tavily_responses])
        print(f"Search results: {search_results}")
        if not search_results:
            raise ValueError("No search results found. Please try again with different queries.")

        prompt_for_asking_to_answer_questions_based_on_search_results = f"""You will be given a conversation between an interviewer and an expert.
The Interview is about {state["company_name"]}'s {self.topic_name}, which is defined as: {self.topic_description}.
You are also given the following background information about the company:
{state["company_background"]}

Your answer should be based only on the search results provided below. Do not add any additional information.
Note that search results may contain irrelevant information so you should use your expertise to filter out the noise and focus on information which is relevant to the topic of the interview and the company being researched.
Here are the search results:
{search_results}"""
        messages = [SystemMessage(content=prompt_for_asking_to_answer_questions_based_on_search_results)] + state["messages"]
        
        answer = await self.llm.ainvoke(messages)
        answer.name = "Expert" 
        
        return {"messages": [answer]}
