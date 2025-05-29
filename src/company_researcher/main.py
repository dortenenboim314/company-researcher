from fastapi import FastAPI
from pydantic import BaseModel
import logging
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from fastapi import Request 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

class ResearchResponse(BaseModel):
    company_background: str
    financial_health: str
    market_position: str
    news: str

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/research", response_model=ResearchResponse)
def search(company_name: str, company_url: str):
    logging.info(f"Received request for company: {company_name}, URL: {company_url}")
    
    return ResearchResponse(
        company_background="Mocked company background information.",
        financial_health="Mocked financial health data.",
        market_position="Mocked market position insights.",
        news="Mocked latest news articles."
    )
