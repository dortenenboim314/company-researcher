from fastapi import Depends, FastAPI
from pydantic import BaseModel
import logging
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from fastapi import Request 
from fastapi.middleware.cors import CORSMiddleware

from company_researcher.models.models import ResearchQuery, ResearchResponse
from company_researcher.mock import mock_research_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
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


@app.get("/api/research", response_model=ResearchResponse)
def get_research(query: ResearchQuery = Depends()):
    logging.info(f"Received request for company: {query.company_name}, URL: {query.company_url}")
    
    return mock_research_response()
