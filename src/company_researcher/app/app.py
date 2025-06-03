from fastapi import Depends, FastAPI
import logging
from fastapi.templating import Jinja2Templates
import os
from fastapi import Request 
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from company_researcher.core.agents import CompanyResearchAgent
from company_researcher.core.api_clients.tavily_client import TavilyClient
from company_researcher.app.schemas.get_research import GetResearchResponse, GetResearchRequest
from company_researcher.config import load_config
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:8000")
allow_origins = list(set([
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    frontend_origin,
]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


config = load_config()
llm = ChatOpenAI(model = config.openai_model, temperature=config.llm_temperature)
tavily_client = TavilyClient()
company_researcher = CompanyResearchAgent(
    llm=llm,
    tavily_client=tavily_client,
    config=config
)


@app.get("/api/research", response_model=GetResearchResponse)
async def get_research(query: GetResearchRequest = Depends()):
    logging.info(f"Received request for company: {query.company_name}, URL: {query.company_url}")
    res = await company_researcher.perform_research(
        company_name=query.company_name,
        company_url=query.company_url
    )
    # time.sleep(35)
    # res = FinalReport(
    #     background_summary="Sample background summary",
    #     financial_health_summary="Sample financial health summary",
    #     market_position_summary="Sample market position summary",
    # )
    
    logging.info(f"response: {res}")
    
    return GetResearchResponse(
        background_summary=res.grounded_information.background,
        financial_health_summary=res.grounded_information.financial_health,
        market_position_summary=res.grounded_information.market_position
    )
    
