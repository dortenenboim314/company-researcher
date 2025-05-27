#!/usr/bin/env python3
"""
Simple end-to-end test for BackgroundAgent with dreamgroup.com
Uses real APIs to test the complete pipeline.
"""

import asyncio
import os
import sys
import time
from typing import Dict, Any, TypedDict, List
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_openai import ChatOpenAI
from src.api_clients.tavily_client import TavilyClient

# Define ResearchState directly here to avoid import issues
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

# Import BackgroundAgent with local ResearchState
from src.agents.BackgroundAgent import BackgroundAgent

# Load environment variables
load_dotenv()

def print_section(title: str, content: str = ""):
    """Helper function to print formatted sections"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")
    if content:
        print(content)

async def test_background_agent_dreamgroup():
    """
    End-to-end test of BackgroundAgent with dreamgroup.com
    """
    print_section("STARTING E2E TEST FOR BACKGROUND AGENT")
    print(f"Target URL: https://dreamgroup.com/")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        return False
    if not tavily_key:
        print("❌ ERROR: TAVILY_API_KEY not found in environment")
        return False
    
    print("✅ API keys found")
    print(f"📝 OpenAI Key: {openai_key[:8]}...{openai_key[-4:]}")
    print(f"📝 Tavily Key: {tavily_key[:8]}...{tavily_key[-4:]}")
    
    try:
        # Initialize components
        print_section("INITIALIZING COMPONENTS")
        
        print("🤖 Creating ChatOpenAI instance...")
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_key
        )
        
        print("🔍 Creating TavilyClient instance...")
        tavily_client = TavilyClient(api_key=tavily_key)
        
        print("🎯 Creating BackgroundAgent instance...")
        config = {
            "max_queries": 3,  # Limit search queries for testing
            "timeout": 120     # 2 minute timeout
        }
        agent = BackgroundAgent(llm=llm, tavily_client=tavily_client, config=config)
        
        print("✅ All components initialized successfully")
        
        # Create research state
        print_section("PREPARING RESEARCH STATE")
        state = ResearchState(
            company_name="Dream Group",
            company_url="https://dreamgroup.com/",
            company_background="",
            financial_data={},
            market_position="",
            recent_news=[],
            final_report="",
            current_step="starting",
            errors=[]
        )
        
        print("✅ Research state created")
        print(f"📋 Company: {state['company_name']}")
        print(f"🌐 URL: {state['company_url']}")
        
        # Run the BackgroundAgent
        print_section("RUNNING BACKGROUND AGENT")
        start_time = time.time()
        
        print("🚀 Starting BackgroundAgent.run()...")
        result = await agent.run(state)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ BackgroundAgent completed in {execution_time:.2f} seconds")
        
        # Display results
        print_section("RESULTS ANALYSIS")
        
        if 'background' in result and 'content' in result['background']:
            background_content = result['background']['content']
            
            print(f"📊 Background content length: {len(background_content)} characters")
            print(f"📝 Word count: {len(background_content.split())} words")
            
            # Check if content mentions the company
            mentions_company = any(term.lower() in background_content.lower() 
                                 for term in ['dream', 'dreamgroup', 'dream group'])
            print(f"🏢 Mentions company: {'✅ Yes' if mentions_company else '❌ No'}")
            
            print_section("FULL BACKGROUND CONTENT")
            print(background_content)
            
            # Basic assertions
            print_section("VALIDATION CHECKS")
            
            checks = [
                ("Background content exists", bool(background_content)),
                ("Content is substantial (>100 chars)", len(background_content) > 100),
                ("Content mentions company", mentions_company),
                ("Execution time reasonable (<300s)", execution_time < 300),
                ("No empty content", background_content.strip() != "")
            ]
            
            all_passed = True
            for check_name, passed in checks:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status}: {check_name}")
                if not passed:
                    all_passed = False
            
            print_section("TEST SUMMARY")
            final_status = "✅ SUCCESS" if all_passed else "❌ FAILED"
            print(f"Test Status: {final_status}")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print(f"Content Length: {len(background_content)} characters")
            
            return all_passed
            
        else:
            print("❌ ERROR: No background content found in result")
            print(f"📋 Result keys: {list(result.keys()) if result else 'None'}")
            print(f"📋 Full result: {result}")
            return False
            
    except Exception as e:
        print_section("ERROR OCCURRED")
        print(f"❌ Exception: {type(e).__name__}")
        print(f"📝 Message: {str(e)}")
        import traceback
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        return False

async def main():
    """Main test runner"""
    print("🧪 BackgroundAgent Simple E2E Test")
    print("=" * 60)
    
    success = await test_background_agent_dreamgroup()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        exit_code = 0
    else:
        print("💥 TESTS FAILED!")
        exit_code = 1
    
    print("=" * 60)
    return exit_code

if __name__ == "__main__":
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
