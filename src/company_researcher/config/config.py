import os
import yaml
from pydantic import BaseModel, Field
import logging

class Config(BaseModel):
    """Configuration for the Company Researcher application."""
    
    # LLM configuration
    openai_model: str = Field(description="The model name for the language model.")
    llm_temperature: float = Field(0, description="Temperature setting for the LLM.")
    
    # Other configurations
    max_searches_per_agent: int = Field(3, description="Maximum number of searches per agent.")

    class Config:
        extra = "forbid"

def load_config() -> Config:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        return Config(**yaml.safe_load(f))