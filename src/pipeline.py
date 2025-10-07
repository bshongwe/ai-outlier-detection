"""Main pipeline orchestration for AI Outlier Detection."""

import pandas as pd
from openai import OpenAI
import logging
import json
import os

from .config import Config, load_config
from .data_preprocessing import DataPreprocessor
from .embedding_generator import EmbeddingGenerator
from .outlier_detectors import (
    OutlierDetectionPipeline, EuclideanDetector,
    MahalanobisDetector, LOFDetector, IsolationForestDetector
)
from .explainer import OutlierExplainer
from .visualizer import OutlierVisualizer

logger = logging.getLogger(__name__)


class AIOutlierDetectionPipeline:
    """Main pipeline for AI-powered outlier detection."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.api.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _initialize_components(self):
        """Initialize all pipeline components."""
        # API client
        self.client = OpenAI(
            base_url=self.config.api.base_url,
            api_key=self.config.api.api_key,
            timeout=self.config.api.timeout
        )

        # Components
        self.preprocessor = DataPreprocessor(
            max_text_length=self.config.processing.max_text_length,
            sample_size=self.config.processing.sample_size
        )

        self.embedding_generator = EmbeddingGenerator(
            client=self.client,
            model_name=self.config.models.embedding,
            batch_size=self.config.processing.batch_size
        )

        # Outlier detectors
        detectors = [
            EuclideanDetector(radius=self.config.outlier_detection.euclidean_radius),
            MahalanobisDetector(confidence_level=self.config.outlier_detection.confidence_level),
            LOFDetector(n_neighbors=self.config.outlier_detection.lof_neighbors),
            IsolationForestDetector(contamination=self.config.outlier_detection.contamination_rate)
        ]

        self.outlier_pipeline = OutlierDetectionPipeline(detectors)

        self.explainer = OutlierExplainer(
            client=self.client,
            model_name=self.config.models.llm
        )

        self.visualizer = OutlierVisualizer()

    def run_full_pipeline(self, save_results: bool = True,
                         output_dir: str = "results") -> Dict:
        """Run the complete outlier detection pipeline."""
        logger.info("Starting AI Outlier Detection Pipeline")

        try:
            # Step 1: Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data")
            df = self.preprocessor.load_newsgroups_data()
            df = self.preprocessor.filter_and_sample_data(df)

            # Step 2: Generate embeddings
            logger.info("Step 2: Generating embeddings")
            df = self.embedding_generator.add_embeddings_to_dataframe(df)

            # Step 3: Run outlier detection
            logger.info("Step 3: Running outlier detection algorithms")
            df = self.outlier_pipeline.run_all_detectors(df)

            # Step 4: Project to 2D for visualization
            logger.info("Step 4: Creating 2D projections")
            df_viz = self.visualizer.project_embeddings_2d(df)

            # Step 5: Generate explanations for outliers
            logger.info("Step 5: Generating explanations")
            outlier_columns = [col for col in df.columns if col.startswith('Outlier_')]
            explanations = {}

            for col in outlier_columns:
                outlier_indices = df[df[col]].index.tolist()
                if outlier_indices:
                    method_explanations = self.explainer.explain_outliers_batch(
                        df, outlier_indices[:5]  # Limit to first 5 for demo
                    )
                    explanations[col] = method_explanations

            # Step 6: Generate visualizations
            logger.info("Step 6: Creating visualizations")
            if save_results:
                os.makedirs(output_dir, exist_ok=True)

                for col in outlier_columns:
                    method_name = col.replace('Outlier_', '')
                    save_path = os.path.join(output_dir, f"{method_name}_outliers.png")
                    self.visualizer.plot_outliers(df_viz, col, save_path=save_path)

                # Comparison plot
                comparison_path = os.path.join(output_dir, "outlier_comparison.png")
                self.visualizer.plot_comparison(df_viz, outlier_columns, save_path=comparison_path)

            # Generate summary
            summary_stats = self.visualizer.generate_summary_stats(df)

            results = {
                'data': df,
                'visualizations': df_viz,
                'explanations': explanations,
                'summary_stats': summary_stats,
                'total_documents': len(df)
            }

            # Save results
            if save_results:
                self._save_results(results, output_dir)

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _save_results(self, results: Dict, output_dir: str):
        """Save pipeline results to files."""
        # Save summary statistics
        with open(os.path.join(output_dir, "summary_stats.json"), 'w') as f:
            json.dump(results['summary_stats'], f, indent=2)

        # Save explanations
        with open(os.path.join(output_dir, "explanations.json"), 'w') as f:
            json.dump(results['explanations'], f, indent=2, default=str)

        # Save processed data
        results['data'].to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)

        logger.info(f"Results saved to {output_dir}")

    def detect_outliers_in_text(self, texts: List[str], categories: List[str] = None) -> Dict:
        """Detect outliers in custom text data."""
        logger.info(f"Processing {len(texts)} custom texts")

        # Create DataFrame
        df = pd.DataFrame({
            'Text': texts,
            'Class Name': categories or ['custom'] * len(texts),
            'Label': range(len(texts))
        })

        # Generate embeddings
        df = self.embedding_generator.add_embeddings_to_dataframe(df)

        # Run outlier detection
        df = self.outlier_pipeline.run_all_detectors(df)

        # Get results
        outlier_columns = [col for col in df.columns if col.startswith('Outlier_')]
        results = {}

        for col in outlier_columns:
            method_name = col.replace('Outlier_', '')
            outlier_indices = df[df[col]].index.tolist()
            results[method_name] = {
                'outlier_indices': outlier_indices,
                'outlier_count': len(outlier_indices),
                'outlier_texts': df.loc[outlier_indices, 'Text'].tolist()
            }

        return results
