import sys
import os

from utils.llm_wrapper import LLMLoggingWrapper
# Add the project root to Python path for direct execution
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  # or use ChatAnthropic, etc.
from datetime import datetime
from dotenv import load_dotenv

from src.api_clients.tavily_client import TavilyClient

load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
print("🔑 Using OpenAI API Key:", OPEN_AI_API_KEY)
print("🔑 Using Tavily API Key:", TAVILY_API_KEY)

# Initialize the LLM (you can change this to your preferred model)
llm = LLMLoggingWrapper(ChatOpenAI(
    model="gpt-4", 
    temperature=0,
))

ChatOpenAI(
    model="gpt-4", 
    temperature=0,
).with_structured_output

# Initialize Tavily client
tavily_client = TavilyClient()

# Define the shared state that all agents will use
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

# Agent 1: Company Background Researcher
def company_background_agent(state: ResearchState) -> ResearchState:
    """
    Researches basic company information, history, and overview
    """
    print(f"🔍 Researching background for {state['company_name']}")
    
    try:
        # Use Tavily API to search for company background
        query = f"{state['company_name']} company background history overview"
        background_data = tavily_client.search(query)
        
        # Process and summarize the background information
        state["company_background"] = process_background_data(background_data)
        state["current_step"] = "background_complete"
        
    except Exception as e:
        state["errors"].append(f"Background research error: {str(e)}")
        state["company_background"] = "Background research failed"
    
    print("✅ Background research completed successfully!")
    return state

# Agent 2: Financial Health Analyzer
def financial_health_agent(state: ResearchState) -> ResearchState:
    """
    Analyzes company's financial health, revenue, profitability
    Uses background context for better analysis
    """
    print(f"💰 Analyzing financial health for {state['company_name']}")
    
    # Search for financial information
    query = f"{state['company_name']} financial health revenue profit earnings"
    financial_data = tavily_client.search(query)
    
    # Analyze financial metrics WITH background context
    state["financial_data"] = analyze_financial_data(financial_data, state["company_background"])
        
    
    print("✅ Financial health analysis completed successfully!")
    return state

# Agent 3: Market Position Researcher
def market_position_agent(state: ResearchState) -> ResearchState:
    """
    Researches company's market position, competitors, industry standing
    Uses background context for better analysis
    """
    print(f"📊 Analyzing market position for {state['company_name']}")
    
    # Search for market position and competitive analysis
    query = f"{state['company_name']} market position competitors industry analysis"
    market_data = tavily_client.search(query)
    
    # Analyze market position WITH background context
    state["market_position"] = analyze_market_position(market_data, state["company_background"])
        
    
    print("✅ Market position analysis completed successfully!")
    return state

# Agent 4: News and Recent Developments Researcher
def news_researcher_agent(state: ResearchState) -> ResearchState:
    """
    Gathers recent news and developments about the company
    Uses background context for better analysis
    """
    print(f"📰 Gathering recent news for {state['company_name']}")
    
    # Search for recent news
    query = f"{state['company_name']} recent news developments 2024 2025"
    news_data = tavily_client.search(query)
    
    # Process and filter relevant news WITH background context
    state["recent_news"] = process_news_data(news_data, state["company_background"])
    
    print("✅ News research completed successfully!")
    return state

# Agent 5: Report Synthesizer
def report_synthesizer_agent(state: ResearchState) -> ResearchState:
    """
    Synthesizes all research into a comprehensive final report
    """
    print(f"📋 Synthesizing final report for {state['company_name']}")
    
    # Combine all research into a comprehensive report
    report = generate_final_report(
        state["company_name"],
        state["company_url"],
        state["company_background"],
        state["financial_data"],
        state["market_position"],
        state["recent_news"]
    )
    
    state["final_report"] = report

    print("✅ Final report generated successfully!")
    return state

# Helper functions for data processing
def process_background_data(data: Dict[str, Any]) -> str:
    """Process and summarize background information using LLM"""
    raw_content = ""
    
    # Extract content from Tavily response
    if 'answer' in data:
        raw_content += data['answer'] + "\n"
    
    if 'results' in data:
        for result in data['results'][:3]:  # Use top 3 results
            raw_content += f"{result.get('content', '')}\n"
    
    if not raw_content.strip():
        return "Background information not available"
    
    # Use LLM to summarize and structure the information
    prompt = f"""
    Analyze the following information about a company and provide a clear, structured summary of the company's background, history, and overview:
    
    {raw_content}
    
    Please provide a concise but comprehensive summary focusing on:
    - Company founding and history
    - Main business activities
    - Key milestones
    - Current status
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Background processing failed: {str(e)}"

def analyze_financial_data(data: Dict[str, Any], background: str) -> Dict[str, Any]:
    """Analyze financial information using LLM with background context"""
    raw_content = ""
    
    if 'answer' in data:
        raw_content += data['answer'] + "\n"
    
    if 'results' in data:
        for result in data['results'][:3]:
            raw_content += f"{result.get('content', '')}\n"
    
    if not raw_content.strip():
        return {"summary": "Financial data not available", "sources": []}
    
    # Use LLM to analyze financial information WITH background context
    prompt = f"""
    Company Background Context:
    {background}
    
    Financial Information to Analyze:
    {raw_content}
    
    Given the company's background and business model, analyze the financial information covering:
    - Revenue trends and financial performance (in context of their business)
    - Profitability and key financial metrics 
    - Financial health and stability
    - How the financials align with their business strategy
    - Any concerning financial indicators
    
    Keep it concise but informative, referencing the business context where relevant.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {
            "summary": response.content,
            "sources": data.get('sources', [])
        }
    except Exception as e:
        return {"summary": f"Financial analysis failed: {str(e)}", "sources": []}

def analyze_market_position(data: Dict[str, Any], background: str) -> str:
    """Analyze market position using LLM"""
    raw_content = ""
    
    if 'answer' in data:
        raw_content += data['answer'] + "\n"
    
    if 'results' in data:
        for result in data['results'][:3]:
            raw_content += f"{result.get('content', '')}\n"
    
    if not raw_content.strip():
        return "Market position data not available"
    
    # Use LLM to analyze market position
    prompt = f"""
    Analyze the following information about a company's market position:
    
    {raw_content}
    
    Please provide an analysis covering:
    - Market share and competitive positioning
    - Key competitors and competitive advantages
    - Industry trends and market dynamics
    - Strengths and weaknesses in the market
    - Future market outlook
    
    Keep it concise but insightful.
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Market position analysis failed: {str(e)}"

def process_news_data(data: Dict[str, Any], background:str) -> List[Dict[str, str]]:
    """Process and filter news data using LLM"""
    if 'results' not in data or not data['results']:
        return []
    
    # Get raw news content
    raw_news = ""
    news_items = []
    
    for item in data['results'][:5]:  # Process top 5 news items
        title = item.get('title', '')
        content = item.get('content', '')
        url = item.get('url', '')
        
        raw_news += f"Title: {title}\nContent: {content}\nURL: {url}\n\n"
        
        news_items.append({
            "title": title,
            "url": url,
            "raw_content": content
        })
    
    if not raw_news.strip():
        return []
    
    # Use LLM to analyze and summarize the news
    prompt = f"""
    Analyze the following recent news about a company and provide key insights:
    
    {raw_news}
    
    For each news item, provide:
    - A concise but informative summary (2-3 sentences)
    - The business impact or significance
    - Keep only the most relevant and recent developments
    
    Format as: Title | Summary | Impact
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        analyzed_content = response.content
        
        # Enhanced news items with LLM analysis
        enhanced_news = []
        for i, item in enumerate(news_items):
            enhanced_news.append({
                "title": item["title"],
                "url": item["url"],
                "summary": f"AI Analysis: {analyzed_content}"  # In practice, you'd parse this better
            })
            
        return enhanced_news[:3]  # Return top 3 analyzed items
        
    except Exception as e:
        # Fallback to simple processing if LLM fails
        fallback_news = []
        for item in news_items:
            fallback_news.append({
                "title": item["title"],
                "url": item["url"],
                "summary": item["raw_content"][:200] + "..."
            })
        return fallback_news

def generate_final_report(company_name: str, company_url: str, background: str, 
                         financial: Dict, market: str, news: List) -> str:
    """Generate comprehensive final report"""
    
    report = f"""
# Company Research Report: {company_name}

## Company Overview
**Website:** {company_url}
**Research Date:** {datetime.now().strftime("%Y-%m-%d")}

## Company Background
{background}

## Financial Health
{financial.get('summary', 'Financial information not available')}

## Market Position
{market}

## Recent News and Developments
"""
    
    for item in news:
        report += f"- **{item['title']}**\n  {item['summary']}\n  Source: {item['url']}\n\n"
    
    return report

# Main function to create and run the research workflow
def create_research_workflow():
    """
    Creates the LangGraph workflow for company research
    """
    
    # Create the state graph
    workflow = StateGraph(ResearchState)
    
    # Add all agent nodes
    workflow.add_node("background_researcher", company_background_agent)
    workflow.add_node("financial_analyzer", financial_health_agent)
    workflow.add_node("market_researcher", market_position_agent)
    workflow.add_node("news_researcher", news_researcher_agent)
    workflow.add_node("report_synthesizer", report_synthesizer_agent)
    
    workflow.add_edge("background_researcher", "financial_analyzer")
    workflow.add_edge("financial_analyzer", "market_researcher")
    workflow.add_edge("market_researcher", "news_researcher")
    workflow.add_edge("news_researcher", "report_synthesizer")
    
    workflow.add_edge("report_synthesizer", END)
    
    # Set the entry point
    workflow.set_entry_point("background_researcher")
    
    # Compile the workflow
    return workflow.compile()

# Main function to run company research
def research_company(workflow, company_name: str, company_url: str) -> str:
    """
    Main function to research a company using the multi-agent system
    
    Args:
        company_name: Name of the company to research
        company_url: Company's website URL
        
    Returns:
        Final research report as string
    """
    
    print(f"🚀 Starting research for {company_name}")
    
    # Initialize the state
    initial_state = {
        "company_name": company_name,
        "company_url": company_url,
        "company_background": "",
        "financial_data": {},
        "market_position": "",
        "recent_news": [],
        "final_report": "",
        "current_step": "starting",
        "errors": []
    }
    
    # Create and run the workflow
    result = workflow.invoke(initial_state)
    
    # Print any errors that occurred
    if result["errors"]:
        print("⚠️ Errors occurred during research:")
        for error in result["errors"]:
            print(f"  - {error}")
    
    print("✅ Research complete!")
    return result["final_report"]

# Example usage
if __name__ == "__main__":
    # Example: Research Apple Inc.
    workflow = create_research_workflow()

    company_name = "Apple Inc."
    company_url = "https://www.apple.com"
    
    report = research_company(workflow, company_name, company_url)
    print("\n" + "="*50)
    print("FINAL RESEARCH REPORT")
    print("="*50)
    print(report)
    
    report_google = research_company(workflow, "Alphabet Inc.", "https://www.google.com")
    print("\n" + "="*50)
    print("FINAL RESEARCH REPORT FOR GOOGLE")
    print("="*50)
    print(report_google)
