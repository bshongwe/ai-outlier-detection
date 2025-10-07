"""LLM-based explanation generation for outliers."""

import pandas as pd
from typing import List, Dict
from openai import OpenAI
import logging
import random

logger = logging.getLogger(__name__)

class OutlierExplainer:
    """Generates human-readable explanations for detected outliers using LLM."""
    
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
    
    def explain_outlier(self, outlier_text: str, normal_texts: List[str], 
                       category: str) -> str:
        """Generate explanation for why a document is an outlier."""
        try:
            prompt = self._create_explanation_prompt(
                outlier_text, normal_texts, category
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            explanation = response.choices[0].message.content
            logger.info(f"Generated explanation for outlier in category {category}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return f"Unable to generate explanation for outlier in {category}"
    
    def _create_explanation_prompt(self, outlier_text: str, normal_texts: List[str], 
                                 category: str) -> str:
        """Create prompt for LLM explanation generation."""
        normal_examples = "\n\n".join([f"Normal example {i+1}: {text[:500]}..." 
                                     for i, text in enumerate(normal_texts)])
        
        prompt = f"""
Category: {category}

Outlier document: {outlier_text[:500]}...

Normal examples from the same category:
{normal_examples}

Analyze why the outlier document is semantically different from the normal examples. 
Provide a concise explanation focusing on content, topic, or style differences.
"""
        return prompt
    
    def explain_outliers_batch(self, df: pd.DataFrame, outlier_indices: List[int], 
                             num_normal_examples: int = 3) -> Dict[int, str]:
        """Generate explanations for multiple outliers."""
        explanations = {}
        
        for idx in outlier_indices:
            try:
                outlier_row = df.loc[idx]
                category = outlier_row["Class Name"]
                outlier_text = outlier_row["Text"]
                
                # Get normal examples from same category
                normal_examples = self._get_normal_examples(
                    df, category, idx, num_normal_examples
                )
                
                explanation = self.explain_outlier(
                    outlier_text, normal_examples, category
                )
                
                explanations[idx] = explanation
                
            except Exception as e:
                logger.error(f"Failed to explain outlier {idx}: {e}")
                explanations[idx] = f"Explanation generation failed for outlier {idx}"
        
        return explanations
    
    def _get_normal_examples(self, df: pd.DataFrame, category: str, 
                           outlier_idx: int, num_examples: int) -> List[str]:
        """Get normal examples from the same category."""
        category_df = df[df["Class Name"] == category]
        normal_df = category_df[category_df.index != outlier_idx]
        
        if len(normal_df) < num_examples:
            num_examples = len(normal_df)
        
        normal_samples = normal_df.sample(n=num_examples, random_state=42)
        return normal_samples["Text"].tolist()