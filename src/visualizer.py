"""Visualization module for outlier detection results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class OutlierVisualizer:
    """Handles visualization of outlier detection results."""
    
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                 random_state: int = 42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        
        # Set consistent plot style
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        plt.rcParams.update({'font.size': 12})
    
    def project_embeddings_2d(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project high-dimensional embeddings to 2D using UMAP."""
        logger.info("Projecting embeddings to 2D using UMAP")
        
        embeddings = np.vstack(df['Embeddings'])
        
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            metric='cosine'
        )
        
        umap_results = reducer.fit_transform(embeddings)
        
        df_umap = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
        df_result = pd.concat([df_umap, df.reset_index(drop=True)], axis=1)
        
        logger.info("UMAP projection completed")
        return df_result
    
    def plot_outliers(self, df: pd.DataFrame, outlier_column: str, 
                     title: str = None, save_path: str = None) -> None:
        """Plot UMAP projection with outliers highlighted."""
        if title is None:
            title = f'Outlier Detection Results: {outlier_column}'
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Plot normal points
        normal_mask = ~df[outlier_column]
        if normal_mask.any():
            scatter = ax.scatter(
                df.loc[normal_mask, 'UMAP1'],
                df.loc[normal_mask, 'UMAP2'],
                c=df.loc[normal_mask, 'Label'],
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            plt.colorbar(scatter, ax=ax)
        
        # Plot outliers
        outlier_mask = df[outlier_column]
        if outlier_mask.any():
            ax.scatter(
                df.loc[outlier_mask, 'UMAP1'],
                df.loc[outlier_mask, 'UMAP2'],
                color='red',
                marker='X',
                s=150,
                label='Outliers',
                edgecolors='black'
            )
        
        ax.set_title(title)
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_comparison(self, df: pd.DataFrame, outlier_columns: List[str], 
                       save_path: str = None) -> None:
        """Plot comparison of multiple outlier detection methods."""
        n_methods = len(outlier_columns)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(outlier_columns):
            ax = axes[i]
            method_name = col.replace('Outlier_', '')
            
            # Plot normal points
            normal_mask = ~df[col]
            if normal_mask.any():
                scatter = ax.scatter(
                    df.loc[normal_mask, 'UMAP1'],
                    df.loc[normal_mask, 'UMAP2'],
                    c=df.loc[normal_mask, 'Label'],
                    cmap='viridis',
                    alpha=0.6,
                    s=30
                )
            
            # Plot outliers
            outlier_mask = df[col]
            if outlier_mask.any():
                ax.scatter(
                    df.loc[outlier_mask, 'UMAP1'],
                    df.loc[outlier_mask, 'UMAP2'],
                    color='red',
                    marker='X',
                    s=100,
                    edgecolors='black'
                )
            
            outlier_count = outlier_mask.sum()
            ax.set_title(f'{method_name}: {outlier_count} outliers')
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, int]:
        """Generate summary statistics for outlier detection results."""
        outlier_columns = [col for col in df.columns if col.startswith('Outlier_')]
        
        stats = {}
        for col in outlier_columns:
            method_name = col.replace('Outlier_', '')
            outlier_count = df[col].sum()
            stats[method_name] = outlier_count
        
        return stats