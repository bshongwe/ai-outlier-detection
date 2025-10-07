"""Data preprocessing module for AI Outlier Detection Pipeline."""

import re
import pandas as pd
from typing import List, Tuple
from sklearn.datasets import fetch_20newsgroups
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data loading and preprocessing operations."""
    
    def __init__(self, max_text_length: int = 2000, sample_size: int = 150):
        self.max_text_length = max_text_length
        self.sample_size = sample_size
    
    def load_newsgroups_data(self, subset: str = "train") -> pd.DataFrame:
        """Load and preprocess 20 Newsgroups dataset."""
        try:
            newsgroups = fetch_20newsgroups(subset=subset)
            logger.info(f"Loaded {len(newsgroups.data)} documents from 20 Newsgroups dataset")
            return self._clean_and_structure_data(newsgroups)
        except Exception as e:
            logger.error(f"Failed to load newsgroups data: {e}")
            raise
    
    def _clean_and_structure_data(self, dataset) -> pd.DataFrame:
        """Clean raw text data and structure into DataFrame."""
        logger.info("Cleaning and structuring data")
        
        # Clean text data
        data = [re.sub(r'[\w\.-]+@[\w\.-]+', '', d) for d in dataset.data]
        data = [re.sub(r'\([^()]*\)', '', d) for d in data]
        data = [d.replace('From: ', '') for d in data]
        data = [d.replace('\nSubject: ', '') for d in data]
        data = [d.replace('Subject: ', '') for d in data]
        data = [d.replace('Lines:', '') for d in data]
        data = [d[:self.max_text_length] for d in data]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=["Text"])
        df["Label"] = dataset.target
        df["Class Name"] = df["Label"].map(dataset.target_names.__getitem__)
        
        logger.info(f"Cleaned {len(df)} documents")
        return df
    
    def filter_and_sample_data(self, df: pd.DataFrame, categories: List[str] = None) -> pd.DataFrame:
        """Filter for specific categories and sample data."""
        if categories is None:
            categories = ["sci.crypt", "sci.electronics", "sci.med", "sci.space"]
        
        # Filter for science categories
        df_filtered = df[df["Class Name"].str.contains("sci")]
        
        # Sample data
        df_sampled = (
            df_filtered.groupby("Class Name", group_keys=False)
            .apply(lambda x: x.sample(min(self.sample_size, len(x)), random_state=42))
            .reset_index(drop=True)
        )
        
        logger.info(f"Filtered and sampled to {len(df_sampled)} documents")
        return df_sampled