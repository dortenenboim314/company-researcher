from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
import logging
from langchain_core.messages import HumanMessage
from company_researcher.agents.BaseAgent import BaseAgent
from company_researcher.api_clients.tavily_client import PageContent, TavilyBatchSearchInput, TavilyClient
from company_researcher.workflow.langgraph_workflow import ResearchState

class FinancialHealthAgent(BaseAgent):
    """
    Agent that gathers and analyzes financial health information for a company.
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
        The main method to run the financial health agent.
        """
        # Use Tavily API to search for company financial information
        logging.info(f"Running FinancialHealthAgent with state: {state}")
        site_content = await self.tavily_client.crawl(
            state['company_url'], 
            max_depth=2, 
            limit=10, 
            instructions="Extract financial information, investor relations data, earnings reports, and financial metrics."
        )
        logging.info(f"Extracted financial site content: {site_content}")
        
        logging.info("Summarizing site content to grounded financial information.")
        grounded_financial_info = self._summarize_to_grounded_financial_info(site_content)
        
        max_queries = self.config.get('max_queries', 4)
        search_input = self._generate_queries_for_missing_financial_info(grounded_financial_info, max_queries)
        search_output_for_missing_info = await self.tavily_client.search(search_input)
        
        # Process and analyze the financial health information
        financial_analysis = self.process_financial_data(
            grounded_financial_info, 
            search_output_for_missing_info,
            state.get('background', {}).get('content', '')
        )
        
        return {
            'financial_health': {
                'content': financial_analysis['summary'],
                'metrics': financial_analysis['metrics'],
                'overall_health_score': financial_analysis['health_score'],
                'risk_factors': financial_analysis['risk_factors']
            }
        }
        
    def process_financial_data(self, grounded_info: str, search_output: List[Any], background_context: str) -> Dict[str, Any]:
        """Process the grounded financial information and search output to create a comprehensive financial health analysis."""
        
        # Combine all search results into a single text for analysis
        search_results_text = ""
        for search_response in search_output:
            if hasattr(search_response, 'answer') and search_response.answer:
                search_results_text += f"Answer: {search_response.answer}\n"
            if hasattr(search_response, 'candidates'):
                for candidate in search_response.candidates[:3]:  # Top 3 results per query
                    search_results_text += f"Title: {candidate.title}\nContent: {candidate.content}\nScore: {candidate.score}\n\n"
        
        prompt = f"""
        You are an expert financial analyst specializing in corporate financial health assessment.
        
        Company Background Context:
        {background_context}
        
        Grounded Financial Information (from company website):
        {grounded_info}
        
        Additional Financial Search Results:
        {search_results_text}
        
        Your task is to create a comprehensive financial health analysis. Consider the search result scores (0-1) as relevance indicators, but use your expertise to evaluate the information.
        
        Please provide a structured analysis with the following components:
        
        1. FINANCIAL SUMMARY: Overall financial health assessment (2-3 paragraphs)
        2. KEY METRICS: Specific financial metrics and ratios found
        3. HEALTH SCORE: Overall financial health rating (Strong/Good/Fair/Weak/Critical) with justification
        4. RISK FACTORS: Identified financial risks and concerns
        
        Focus on:
        - Revenue trends and growth patterns
        - Profitability metrics (margins, EBITDA, net income)
        - Financial stability indicators (debt levels, liquidity, cash flow)
        - Recent financial performance vs historical
        - Any red flags or concerning trends
        
        Format your response as a JSON object with keys: summary, metrics, health_score, risk_factors
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Try to parse as JSON, fallback to structured text if needed
        try:
            import json
            # Try to extract JSON from response
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            parsed_response = json.loads(content)
            return parsed_response
        except:
            # Fallback to structured parsing
            return self._parse_financial_response_fallback(response.content)
        
    def _parse_financial_response_fallback(self, content: str) -> Dict[str, Any]:
        """Fallback method to parse financial analysis if JSON parsing fails."""
        return {
            "summary": content,
            "metrics": "Metrics analysis included in summary",
            "health_score": "Assessment included in summary", 
            "risk_factors": "Risk factors included in summary"
        }
        
    def _summarize_to_grounded_financial_info(self, site_content: List[PageContent]) -> str:
        """Summarize the crawled site content to grounded financial information."""
        
        if not site_content:
            logging.error("No financial content found on the site.")
            raise ValueError("No financial content found on the site.")
        
        # Use LLM to summarize the crawled financial content
        content_parts = []
        for item in site_content:
            url = item.url.strip()
            text = item.raw_content.strip()
            content_parts.append(f"URL: {url}\nContent:\n{text}")
        
        content = '\n\n'.join(content_parts)
        logging.info(f"Summarizing financial site content, number of parts: {len(content_parts)}")
        
        prompt = f"""
        You are an expert financial analyst summarizing company financial information.
        Your summary should be grounded in the provided content. Do not make assumptions or add information not present in the content.    
        Please provide a concise but comprehensive summary focusing on:
        - Financial performance metrics (revenue, profit, growth)
        - Financial position (assets, liabilities, cash flow)
        - Recent financial results and trends
        - Any financial guidance or forecasts mentioned
        - Debt levels and financial stability indicators
        
        Content to analyze:
        {content}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _generate_queries_for_missing_financial_info(self, grounded_info: str, max_queries: int) -> TavilyBatchSearchInput:
        """Generate queries for any missing financial information based on the grounded information."""
        
        prompt = f"""
        You are an expert in generating search queries for missing financial information.
        Based on the following grounded financial information, generate search queries to find any missing details about the company's financial health.
        Do not repeat information already present in the grounded information.
        Focus specifically on financial data such as:
        - Recent earnings reports and financial statements
        - Revenue and profit trends
        - Financial ratios and metrics
        - Debt levels and financial stability
        - Cash flow and liquidity
        - Financial performance analysis
        
        The queries should be concise and specific, and should not exceed 15 words each.
        You should generate 0-{max_queries} queries.
        
        Current grounded financial information:
        {grounded_info}
        """
        
        response = self.llm.with_structured_output(TavilyBatchSearchInput).invoke([HumanMessage(content=prompt)])
        return response
