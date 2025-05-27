from abc import ABC, abstractmethod
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from company_researcher.api_clients.tavily_client import TavilyClient


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the company_researcher project.

    All agents must accept an LLM, a TavilyClient, and a configuration dict,
    and implement an asynchronous `run` method to perform their task.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tavily_client: TavilyClient,
        config: Dict[str, Any]
    ):
        """
        Initializes the BaseAgent with core dependencies.

        Args:
            llm: An instance of ChatOpenAI for language model calls.
            tavily_client: An instance of TavilyClient for data retrieval.
            config: A configuration dictionary with agent-specific settings.
        """
        self.llm = llm
        self.tavily_client = tavily_client
        self.config = config

    @abstractmethod
    async def run(self, state: Any) -> Any:
        """
        Main entry point for the agent. Must be implemented by subclasses.

        Args:
            state: The current research state or input data for the agent.

        Returns:
            The updated state or result after agent execution.
        """
        raise NotImplementedError("Subclasses must implement the run method.")