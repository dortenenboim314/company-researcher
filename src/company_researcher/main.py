from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI()

class Request(BaseModel):
    company_name: str
    company_url: str

class Response(BaseModel):
    company_background: str
    financial_health: str
    market_position: str
    news: str

@app.get("/searchCompany", response_model=Response)
def search(company_name: str, company_url: str):
    logging.info(f"Received request for company: {company_name}, URL: {company_url}")
    
    return Response(
        company_background="Mocked company background information.",
        financial_health="Mocked financial health data.",
        market_position="Mocked market position insights.",
        news="Mocked latest news articles."
    )