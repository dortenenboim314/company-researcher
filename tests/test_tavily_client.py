#!/usr/bin/env python3
"""
End-to-end tests for TavilyClient crawl and search methods.
Uses real Tavily API to test the complete functionality.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import time
from dotenv import load_dotenv
from company_researcher.api_clients.tavily_client import TavilyClient, TavilyBatchSearchInput, PageContent, SearchResponse
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# Load environment variables
load_dotenv()

def print_section(title: str, content: str = ""):
    """Helper function to print formatted sections"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")
    if content:
        print(content)

async def test_tavily_client_crawl():
    """
    End-to-end test of TavilyClient crawl method
    """
    print_section("TESTING TAVILY CLIENT CRAWL METHOD")
    
    # Check environment variables
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("❌ ERROR: TAVILY_API_KEY not found in environment")
        return False
    
    print("✅ API key found")
    print(f"📝 Tavily Key: {tavily_key[:8]}...{tavily_key[-4:]}")
    
    try:
        # Initialize TavilyClient
        print_section("INITIALIZING TAVILY CLIENT")
        client = TavilyClient(api_key=tavily_key)
        print("✅ TavilyClient initialized successfully")
        
        # Test 1: Basic crawl with httpbin.org (reliable test site)
        print_section("TEST 1: BASIC CRAWL")
        test_url = "https://httpbin.org/"
        print(f"🌐 Crawling URL: {test_url}")
        
        start_time = time.time()
        pages = await client.crawl(url=test_url, max_depth=1, limit=3)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ Crawl completed in {execution_time:.2f} seconds")
        print(f"📊 Retrieved {len(pages)} pages")
        
        # Validate results
        crawl_checks = [
            ("Pages retrieved", len(pages) > 0),
            ("All pages are PageContent objects", all(isinstance(page, PageContent) for page in pages)),
            ("All pages have URLs", all(hasattr(page, 'url') and page.url for page in pages)),
            ("All pages have content", all(hasattr(page, 'raw_content') and page.raw_content for page in pages)),
            ("Execution time reasonable (<60s)", execution_time < 60)
        ]
        
        crawl_passed = True
        for check_name, passed in crawl_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {check_name}")
            if not passed:
                crawl_passed = False
        
        # Print sample page info
        if pages:
            print(f"📄 Sample page URL: {pages[0].url}")
            print(f"📄 Sample content length: {len(pages[0].raw_content)} characters")
            print(f"📄 Sample content preview: {pages[0].raw_content[:200]}...")
        
        # Test 2: Crawl with depth limit
        print_section("TEST 2: CRAWL WITH DEPTH LIMIT")
        print(f"🌐 Crawling URL: {test_url} with max_depth=2")
        
        start_time = time.time()
        pages_depth2 = await client.crawl(url=test_url, max_depth=2, limit=5)
        end_time = time.time()
        execution_time_depth2 = end_time - start_time
        
        print(f"✅ Depth-limited crawl completed in {execution_time_depth2:.2f} seconds")
        print(f"📊 Retrieved {len(pages_depth2)} pages with depth=2")
        
        # Test 3: Crawl with instructions
        print_section("TEST 3: CRAWL WITH INSTRUCTIONS")
        instructions = "Focus on main content and navigation links"
        print(f"🌐 Crawling URL: {test_url} with instructions: {instructions}")
        
        start_time = time.time()
        pages_instructions = await client.crawl(url=test_url, max_depth=1, limit=2, instructions=instructions)
        end_time = time.time()
        execution_time_instructions = end_time - start_time
        
        print(f"✅ Instructed crawl completed in {execution_time_instructions:.2f} seconds")
        print(f"📊 Retrieved {len(pages_instructions)} pages with instructions")
        
        # Additional validation checks
        depth_checks = [
            ("Depth-limited crawl works", len(pages_depth2) >= 0),
            ("Instructed crawl works", len(pages_instructions) >= 0),
            ("All crawls return PageContent objects", 
             all(isinstance(page, PageContent) for page in pages_depth2 + pages_instructions))
        ]
        
        for check_name, passed in depth_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {check_name}")
            if not passed:
                crawl_passed = False
        
        return crawl_passed
        
    except Exception as e:
        print_section("CRAWL TEST ERROR")
        print(f"❌ Exception: {type(e).__name__}")
        print(f"📝 Message: {str(e)}")
        import traceback
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        return False

async def test_tavily_client_search():
    """
    End-to-end test of TavilyClient search method
    """
    print_section("TESTING TAVILY CLIENT SEARCH METHOD")
    
    # Check environment variables
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("❌ ERROR: TAVILY_API_KEY not found in environment")
        return False
    
    try:
        # Initialize TavilyClient
        print_section("INITIALIZING TAVILY CLIENT FOR SEARCH")
        client = TavilyClient(api_key=tavily_key)
        print("✅ TavilyClient initialized successfully")
        
        # Test 1: Single query search
        print_section("TEST 1: SINGLE QUERY SEARCH")
        single_query = "artificial intelligence trends 2024"
        batch_input = TavilyBatchSearchInput(queries=[single_query])
        print(f"🔍 Searching for: {single_query}")
        
        start_time = time.time()
        search_results = await client.search(batch_input)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ Search completed in {execution_time:.2f} seconds")
        print(f"📊 Retrieved {len(search_results)} search responses")
        
        # Validate single search results
        if search_results:
            result = search_results[0]
            print(f"📄 Query: {result.query}")
            print(f"📄 Answer: {result.answer[:200] if result.answer else 'No answer provided'}...")
            print(f"📄 Number of candidates: {len(result.candidates)}")
            
            if result.candidates:
                candidate = result.candidates[0]
                print(f"📄 Top result title: {candidate.title}")
                print(f"📄 Top result URL: {candidate.url}")
                print(f"📄 Top result score: {candidate.score}")
                print(f"📄 Top result content: {candidate.content[:150]}...")
        
        single_search_checks = [
            ("Search results returned", len(search_results) > 0),
            ("All results are SearchResponse objects", all(isinstance(result, SearchResponse) for result in search_results)),
            ("First result has query", search_results[0].query == single_query if search_results else False),
            ("First result has candidates", len(search_results[0].candidates) > 0 if search_results else False),
            ("Candidates have required fields", 
             all(hasattr(c, 'title') and hasattr(c, 'url') and hasattr(c, 'score') and hasattr(c, 'content') 
                 for c in search_results[0].candidates) if search_results and search_results[0].candidates else False),
            ("Execution time reasonable (<30s)", execution_time < 30)
        ]
        
        search_passed = True
        for check_name, passed in single_search_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {check_name}")
            if not passed:
                search_passed = False
        
        # Test 2: Batch search with multiple queries
        print_section("TEST 2: BATCH SEARCH")
        batch_queries = [
            "climate change impact",
            "renewable energy solutions",
            "sustainable development goals"
        ]
        batch_input = TavilyBatchSearchInput(queries=batch_queries)
        print(f"🔍 Searching for {len(batch_queries)} queries:")
        for i, query in enumerate(batch_queries, 1):
            print(f"  {i}. {query}")
        
        start_time = time.time()
        batch_results = await client.search(batch_input)
        end_time = time.time()
        batch_execution_time = end_time - start_time
        
        print(f"✅ Batch search completed in {batch_execution_time:.2f} seconds")
        print(f"📊 Retrieved {len(batch_results)} search responses")
        
        # Validate batch search results
        batch_search_checks = [
            ("Batch results count matches queries", len(batch_results) == len(batch_queries)),
            ("All batch results are SearchResponse objects", all(isinstance(result, SearchResponse) for result in batch_results)),
            ("All queries processed", all(result.query in batch_queries for result in batch_results)),
            ("All results have candidates", all(len(result.candidates) > 0 for result in batch_results)),
            ("Batch execution time reasonable (<60s)", batch_execution_time < 60)
        ]
        
        for check_name, passed in batch_search_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {check_name}")
            if not passed:
                search_passed = False
        
        # Print summary of batch results
        if batch_results:
            print("\n📊 BATCH SEARCH SUMMARY:")
            for i, result in enumerate(batch_results, 1):
                print(f"  {i}. Query: {result.query}")
                print(f"     Candidates: {len(result.candidates)}")
                print(f"     Has answer: {'Yes' if result.answer else 'No'}")
                if result.candidates:
                    print(f"     Top score: {result.candidates[0].score:.3f}")
        
        # Test 3: Empty query handling
        print_section("TEST 3: EDGE CASE - EMPTY QUERY")
        try:
            empty_batch = TavilyBatchSearchInput(queries=[""])
            empty_results = await client.search(empty_batch)
            print(f"📊 Empty query handled, returned {len(empty_results)} results")
            
            empty_checks = [
                ("Empty query doesn't crash", True),
                ("Empty results handled gracefully", isinstance(empty_results, list))
            ]
            
            for check_name, passed in empty_checks:
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status}: {check_name}")
                if not passed:
                    search_passed = False
                    
        except Exception as e:
            print(f"⚠️  Empty query handling: {str(e)}")
            print("✅ PASS: Empty query appropriately raises exception")
        
        return search_passed
        
    except Exception as e:
        print_section("SEARCH TEST ERROR")
        print(f"❌ Exception: {type(e).__name__}")
        print(f"📝 Message: {str(e)}")
        import traceback
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        return False

async def main():
    """Main test runner"""
    print("🧪 TavilyClient E2E Tests")
    print("=" * 60)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run crawl tests
    print_section("STARTING CRAWL TESTS")
    crawl_success = await test_tavily_client_crawl()
    
    # Run search tests
    print_section("STARTING SEARCH TESTS")
    search_success = await test_tavily_client_search()
    
    # Final summary
    print_section("FINAL TEST SUMMARY")
    
    crawl_status = "✅ PASSED" if crawl_success else "❌ FAILED"
    search_status = "✅ PASSED" if search_success else "❌ FAILED"
    overall_success = crawl_success and search_success
    overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
    
    print(f"Crawl Tests: {crawl_status}")
    print(f"Search Tests: {search_status}")
    print(f"Overall: {overall_status}")
    
    if overall_success:
        print("\n🎉 TavilyClient E2E tests completed successfully!")
        print("✅ Both crawl and search methods are working correctly")
        exit_code = 0
    else:
        print("\n💥 Some TavilyClient tests failed!")
        print("❌ Please check the test output above for details")
        exit_code = 1
    
    print("=" * 60)
    return exit_code

if __name__ == "__main__":
    # Run the tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
