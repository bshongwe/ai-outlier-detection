"""Configuration management for AI Outlier Detection Pipeline."""

import os
import yaml
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class APIConfig(BaseModel):
    base_url: str = "https://api.studio.nebius.com/v1/"
    api_key: str = ""
    timeout: int = 30
    max_retries: int = 3

class ModelConfig(BaseModel):
    embedding: str = "BAAI/bge-en-icl"
    llm: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

class ProcessingConfig(BaseModel):
    batch_size: int = 50
    max_text_length: int = 2000
    sample_size: int = 150

class OutlierConfig(BaseModel):
    euclidean_radius: float = 0.55
    contamination_rate: float = 0.05
    lof_neighbors: int = 20
    confidence_level: float = 0.99

class Config(BaseModel):
    api: APIConfig = APIConfig()
    models: ModelConfig = ModelConfig()
    processing: ProcessingConfig = ProcessingConfig()
    outlier_detection: OutlierConfig = OutlierConfig()

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file and environment variables."""
    config_dict = {}
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    
    # Override with environment variables
    if os.getenv('NEBIUS_API_KEY'):
        config_dict.setdefault('api', {})['api_key'] = os.getenv('NEBIUS_API_KEY')
    
    return Config(**config_dict)