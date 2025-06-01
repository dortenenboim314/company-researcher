
from typing import TypedDict
from langchain_openai import ChatOpenAI
from company_researcher.core.agents import BackgroundAgent, FinancialHealthAgent, MarketPositionAgent, NewsAgent
from company_researcher.core.api_clients.tavily_client import TavilyBatchSearchInput, TavilyClient
from company_researcher.core.utils.llm_wrapper import LLMLoggingWrapper
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState


class TopicResearchInput(TypedDict):
    company_name: str
    company_background: str

class TopicResearchOutput(TypedDict):
    result: str
    
class TopicResearchState(MessagesState):
    company_name: str
    company_background: str
    result: str
    
    
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
        
        self.graph.add_node("ask_question", self.ask_question)
        self.graph.add_node("search_web_and_answer", self.search_web_and_answer)
        self.graph.add_node("summarize_results", self.summarize_results)
        
        self.graph.add_edge(START, "ask_question")
        self.graph.add_conditional_edges("ask_question", self.route_to_search_or_summarize, ["search_web_and_answer", "summarize_results"])
        self.graph.add_edge("search_web_and_answer", "ask_question")
        self.graph.add_edge("summarize_results", END)
        
    def summarize_results(self, state: TopicResearchState) -> TopicResearchState:
        """Summarize the results of the research.

        Args:
            state (TopicResearchState): The current state of the research.

        Returns:
            TopicResearchState: The updated state with the summary.
        """
        # This should be replaced with the actual logic to summarize the results
        prompt_for_summarizing_results = "" # should be implemented, basically telling the LLM to summarize the conversation between the user and the Interviewer into a concise, reliable summary. do not add any additional information, just make a report of the topic research etc..
        
        summary = self.llm.invoke([SystemMessage(content=prompt_for_summarizing_results)] + state.messages)
        
        return {"result": summary.content}
        
    def route_to_search_or_summarize(self, state: TopicResearchState) -> str:
        """Decide whether to search the web or summarize based on the current state.

        Args:
            state (TopicResearchState): The current state of the research.

        Returns:
            str: The next step in the workflow, either "search_web" or "summarize_results".
        """
        # count the number of messages sent by the expert
        expert_messages = [msg for msg in state.messages if msg.name == "Expert"]
        if len(expert_messages) >= self.max_steps:
            return "summarize_results"
        else:
            return "search_web"

    def ask_question(self, state: TopicResearchState) -> TopicResearchState:
        """Prompt an LLM to be a researcher and to ask questions regarding the topic.

        Args:
            state (TopicResearchState): _description_

        Returns:
            TopicResearchState: _description_
        """
        prompt_for_researcher = "" # This should be replaced with the actual logic to generate a question based on the topic
        messages = state.messages
        
        question = self.llm.invoke([SystemMessage(content=prompt_for_researcher)] + messages)
        question.name = "Interviewer"
        
        return {"messagess": [question]}
    
    async def search_web_and_answer(self, state: TopicResearchState) -> TopicResearchState:
        """Search the web for information related to the topic.

        Args:
            state (TopicResearchState): _description_

        Returns:
            TopicResearchState: _description_
        """
        # This should be replaced with the actual logic to search the web
        
        prompt_asking_for_search_queries = "" # should be implemented
        
        search_queries = self.llm.with_structured_output(TavilyBatchSearchInput).invoke(prompt_asking_for_search_queries)
        tavily_responses = await self.tavily_client.search(search_queries)
        
        search_results = []
        for response in tavily_responses:
            info = []
            if response.query:
                info.append(f"Query: {response.query}")
            if response.answer:
                info.append(f"Snippet: {response.answer}")
            search_results.append('\n'.join(info))

        prompt_for_asking_to_answer_questions_based_on_search_results = ""  # should be implemented, of course we should inject the search results into the prompt
        
        messages = [SystemMessage(content=prompt_for_asking_to_answer_questions_based_on_search_results)] + state.messages
        
        answer = self.llm.invoke(messages)
        answer.name = "Expert" 
        
        return {"messages": [answer]}