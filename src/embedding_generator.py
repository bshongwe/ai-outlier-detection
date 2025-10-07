"""Embedding generation module using OpenAI-compatible APIs."""

import numpy as np
import pandas as pd
from typing import List
from openai import OpenAI
from tqdm import tqdm
import logging
import time

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings using OpenAI-compatible API."""
    
    def __init__(self, client: OpenAI, model_name: str, batch_size: int = 50):
        self.client = client
        self.model_name = model_name
        self.batch_size = batch_size
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i+self.batch_size]
            
            try:
                embeddings = self._process_batch(batch)
                all_embeddings.extend(embeddings)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size}: {e}")
                # Add None placeholders for failed batch
                all_embeddings.extend([None] * len(batch))
        
        # Filter out None values
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]
        logger.info(f"Successfully generated {len(valid_embeddings)} embeddings")
        
        return all_embeddings
    
    def _process_batch(self, batch: List[str]) -> List[np.ndarray]:
        """Process a single batch of texts."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=batch
        )
        
        return [np.array(item.embedding) for item in response.data]
    
    def add_embeddings_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add embeddings to DataFrame and remove failed entries."""
        texts = df['Text'].tolist()
        embeddings = self.generate_embeddings(texts)
        
        df['Embeddings'] = embeddings
        df_clean = df.dropna(subset=['Embeddings']).reset_index(drop=True)
        
        logger.info(f"Added embeddings to {len(df_clean)} documents")
        return df_clean