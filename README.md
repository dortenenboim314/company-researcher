# Company Researcher 🔎

An AI-powered company research tool that provides comprehensive analysis of businesses using advanced LangGraph agents and web search capabilities. The application automatically gathers background information, analyzes financial health, evaluates market position, and identifies positive and negative aspects of any company.

## Project Overview

Company Researcher is an intelligent research assistant that automates the process of company analysis. Simply provide a company name and URL, and the system will:

- **Background Research**: Gather company history, industry information, founding details, mission/vision, milestones, and employee estimates
- **Financial Health Analysis**: Analyze revenue, expenses, profitability, and financial trends
- **Market Position Evaluation**: Assess competitors, market share, and industry positioning
- **Comprehensive Reporting**: Generate structured reports with positive and negative aspects

The application uses multiple specialized AI agents working in parallel to provide thorough, well-researched company insights.

## Architecture Summary

The system follows a multi-agent architecture powered by LangGraph:

### Agent Flow
1. **CompanyResearchAgent** - Acts as the main orchestrator: it coordinates all agents, collects their outputs, and compiles the final research report.
2. **BackgroundAgent** - Specializes in gathering fundamental company information
3. **Financial Health Agent** - Focuses on financial data and health indicators
4. **Market Position Agent** - Analyzes competitive landscape and market positioning
5. **Results Summarization** - Consolidates findings into structured output

### System Components
- **FastAPI Backend** - RESTful API serving research requests
- **LangGraph Orchestration** - Manages agent workflows and state transitions
- **Tavily API Integration** - Performs intelligent web searches
- **OpenAI LLM Processing** - Analyzes and synthesizes research data
- **MongoDB Logging** - Stores research results and maintains history
- **Web UI** - Simple, responsive interface for user interaction

### Data Flow
Input (Company Name + URL) → Company Researcher → MongoDB Logging → UI Display

#### CompanyResearcher:
<img src="docs/company_researcher.jpeg" alt="Company Researcher" width="400"/>

#### Financial Researcher & Market Researcher
<img src="docs/topic_researcher.jpeg" alt="Topic Researcher" width="400"/>


## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **LangGraph** - Agent workflow orchestration
- **LangChain** - LLM integration and processing
- **OpenAI GPT** - Language model for analysis and synthesis
- **Pydantic** - Data validation and serialization

### Data & Search
- **Tavily API** - Intelligent web search and content extraction
- **MongoDB Atlas** - Cloud database for result logging
- **PyMongo** - MongoDB Python driver

### Frontend
- **HTML/CSS/JavaScript** - Lightweight, responsive web interface

### Deployment & Infrastructure
- **AWS Elastic Beanstalk** - Cloud application platform
- **Gunicorn + Uvicorn** - ASGI server stack

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd company-researcher
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the example environment file and configure your API keys:

```bash
cp .env-example .env
```

Edit `.env` with your credentials:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
MONGO_URI=your_mongodb_connection_string_here
```

### 4. Required API Keys

#### OpenAI API Key
- Sign up at [OpenAI Platform](https://platform.openai.com/)
- Create an API key in your dashboard
- Ensure you have sufficient credits for GPT model usage

#### Tavily API Key
- Register at [Tavily](https://tavily.com/)
- Obtain your API key from the dashboard

#### MongoDB Atlas Setup
- Create a free account at [MongoDB Atlas](https://www.mongodb.com/atlas)
- Create a new cluster
- Set up database user and network access
- Get your connection string in format:
```
mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>?retryWrites=true&w=majority
```

### 5. Configuration
The application uses `src/company_researcher/config/config.yaml` for settings. Default configuration:
```yaml
openai_model: "gpt-4"
llm_temperature: 0
max_searches_per_agent: 3
```

## Run Locally

### Start the Application
```bash
uvicorn src.company_researcher.app.app:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/research

### Using the Web Interface
1. Navigate to http://localhost:8000
2. Enter the company name (e.g., "Tesla")
3. Enter the company URL (e.g., "https://www.tesla.com")
4. Click "Research" to start the analysis
5. Wait for the multi-stage research process to complete
6. Review the comprehensive report with background, financial health, market position, and key insights

## Deploy Instructions

### AWS Elastic Beanstalk Deployment

#### 1. Install EB CLI
```bash
pip install awsebcli
```

#### 2. Initialize Elastic Beanstalk
```bash
eb init
```
- Select your AWS region
- Choose Python platform
- Select the latest Python 3.x version

#### 3. Create Environment
```bash
eb create company-researcher-env
```

#### 4. Configure Environment Variables
Set your environment variables in the EB console or via CLI:
```bash
eb setenv OPENAI_API_KEY=your_key TAVILY_API_KEY=your_key MONGO_URI=your_uri
```

#### 5. Deploy
```bash
eb deploy
```

### Deployment Files
The repository includes deployment configuration:
- **Procfile**: Gunicorn server configuration for Elastic Beanstalk
- **application.py**: WSGI entry point
- **requirements.txt**: Python dependencies

### MongoDB Atlas Configuration
1. Whitelist your Elastic Beanstalk IP ranges in MongoDB Atlas Network Access
2. Create a dedicated database user for the application
3. Use the connection string in your environment variables

## Example Usage

### Sample Research Process
1. **Input**: Company Name: "Tesla", URL: "https://www.tesla.com"
2. **Processing**: 
   - Background research collects founding information, mission, industry details
   - Financial agent analyzes revenue, profitability, and financial trends
   - Market position agent evaluates competitors and market share
3. **Output**: Structured report with:
   - **Background**: Founded in 2003, electric vehicle manufacturer, sustainable energy focus
   - **Financial Health**: Revenue growth, profitability trends, investment patterns
   - **Market Position**: EV market leader, competition from traditional automakers
   - **Positive Aspects**: Innovation, brand strength, technological advancement
   - **Negative Aspects**: Production challenges, regulatory risks, market volatility

### API Usage
```bash
curl "http://localhost:8000/api/research?company_name=Tesla&company_url=https://www.tesla.com"
```

Response format:
```json
{
  "background_summary": "Detailed company background...",
  "financial_health_summary": "Financial analysis...",
  "market_position_summary": "Market position evaluation...",
  "positive_aspects": ["Innovation leadership", "Strong brand recognition"],
  "negative_aspects": ["Production scaling challenges", "Regulatory dependencies"]
}
```

## Project Structure

```
company-researcher/
├── src/company_researcher/
│   ├── app/                          # FastAPI application
│   │   ├── app.py                   # Main FastAPI app with routes
│   │   ├── schemas/                 # Pydantic models
│   │   └── templates/               # Jinja2 HTML templates
│   ├── config/                      # Configuration management
│   │   ├── config.py               # Configuration loader
│   │   └── config.yaml             # Application settings
│   └── core/                       # Core business logic
│       ├── agents/                 # LangGraph agents
│       │   ├── company_researcher.py    # Main research orchestrator
│       │   ├── background.py           # Background research agent
│       │   ├── news.py                 # News research agent
│       │   ├── research_topic_interviewer.py  # Topic-specific research
│       │   └── prompts/               # Agent prompts and templates
│       ├── api_clients/            # External API integrations
│       │   └── tavily_client.py    # Tavily search client
│       └── db/                     # Database operations
│           └── mongo_logger.py     # MongoDB logging utilities
├── application.py                  # WSGI entry point for deployment
├── Procfile                       # Elastic Beanstalk process configuration
├── requirements.txt               # Python dependencies
├── .env-example                   # Environment variables template
└── README.md                      # This file
```

### Agent Architecture

#### CompanyResearchAgent
- **Purpose**: Main orchestrator coordinating all research activities
- **Flow**: Background Research → Parallel Financial & Market Analysis → Results Synthesis
- **Output**: Structured company research report

#### BackgroundAgent
- **Purpose**: Fundamental company information gathering
- **Research Areas**: History, industry, founding, mission, milestones, size
- **Search Strategy**: Company websites, official sources, news articles

#### TopicResearchAgent (Financial Health)
- **Purpose**: Financial analysis and health assessment
- **Research Areas**: Revenue, expenses, profitability, financial trends
- **Search Strategy**: Financial reports, earnings news, analyst reports

#### TopicResearchAgent (Market Position)
- **Purpose**: Competitive landscape and market analysis
- **Research Areas**: Competitors, market share, industry trends, positioning
- **Search Strategy**: Industry reports, competitive analysis, market research

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues:
1. Check the existing GitHub issues
2. Create a new issue with detailed information
3. Include logs and error messages when applicable

---

**Company Researcher** - Intelligent company analysis powered by AI agents 🚀
