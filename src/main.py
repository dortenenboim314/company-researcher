import asyncio
from models.company import CompanyInput
from workflow.langgraph_workflow import run_research_graph

def get_company_details():
    name = input("Enter company name to research: ").strip()
    url = input("Enter company URL: ").strip()
    return CompanyInput(name=name, url=url)

async def main():
    company = get_company_details()
    results = await run_research_graph(company)

    print("\n=== Research Results ===")
    for section, data in results.items():
        print(f"\n## {section.capitalize()}\n")
        print(data)

if __name__ == "__main__":
    asyncio.run(main())