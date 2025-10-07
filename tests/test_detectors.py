"""Tests for outlier detection algorithms."""

import pytest
import pandas as pd

from src.outlier_detectors import (
    EuclideanDetector, MahalanobisDetector,
    LOFDetector, IsolationForestDetector,
    OutlierDetectionPipeline
)

@pytest.fixture

def sample_dataframe():
    """Create a sample DataFrame with embeddings."""
    np.random.seed(42)
    embeddings = [np.random.rand(10) for _ in range(20)]
    return pd.DataFrame({
        'Text': [f'Document {i}' for i in range(20)],
        'Embeddings': embeddings,
        'Label': list(range(20)),
        'Class Name': ['class_a'] * 10 + ['class_b'] * 10
    })


class TestEuclideanDetector:
    """Test Euclidean distance-based outlier detection."""

    def test_euclidean_detector_initialization(self):
        """Test detector initialization."""
        detector = EuclideanDetector(radius=0.5)
        assert detector.radius == 0.5
        assert detector.name == "Euclidean"

    def test_euclidean_detection(self, sample_dataframe):
        """Test Euclidean outlier detection."""
        detector = EuclideanDetector(radius=2.0)
        result_df = detector.detect(sample_dataframe)

        assert 'Outlier_Euclidean' in result_df.columns
        assert result_df['Outlier_Euclidean'].dtype == bool


class TestMahalanobisDetector:
    """Test Mahalanobis distance-based outlier detection."""

    def test_mahalanobis_detector_initialization(self):
        """Test detector initialization."""
        detector = MahalanobisDetector(confidence_level=0.95)
        assert detector.confidence_level == 0.95
        assert detector.name == "Mahalanobis"

    def test_mahalanobis_detection(self, sample_dataframe):
        """Test Mahalanobis outlier detection."""
        detector = MahalanobisDetector(confidence_level=0.90)
        result_df = detector.detect(sample_dataframe)

        assert 'Outlier_Mahalanobis' in result_df.columns
        assert result_df['Outlier_Mahalanobis'].dtype == bool


class TestLOFDetector:
    """Test Local Outlier Factor detection."""

    def test_lof_detector_initialization(self):
        """Test detector initialization."""
        detector = LOFDetector(n_neighbors=10)
        assert detector.n_neighbors == 10
        assert detector.name == "LOF"

    def test_lof_detection(self, sample_dataframe):
        """Test LOF outlier detection."""
        detector = LOFDetector(n_neighbors=5)
        result_df = detector.detect(sample_dataframe)

        assert 'Outlier_LOF' in result_df.columns
        assert result_df['Outlier_LOF'].dtype == bool


class TestIsolationForestDetector:
    """Test Isolation Forest detection."""

    def test_isolation_forest_initialization(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(contamination=0.1)
        assert detector.contamination == 0.1
        assert detector.name == "IsolationForest"

    def test_isolation_forest_detection(self, sample_dataframe):
        """Test Isolation Forest outlier detection."""
        detector = IsolationForestDetector(contamination=0.1)
        result_df = detector.detect(sample_dataframe)

        assert 'Outlier_IsolationForest' in result_df.columns
        assert result_df['Outlier_IsolationForest'].dtype == bool


class TestOutlierDetectionPipeline:
    """Test the complete outlier detection pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with multiple detectors."""
        detectors = [
            EuclideanDetector(),
            IsolationForestDetector()
        ]
        pipeline = OutlierDetectionPipeline(detectors)

        assert len(pipeline.detectors) == 2
        assert pipeline.detectors[0].name == "Euclidean"
        assert pipeline.detectors[1].name == "IsolationForest"

    def test_pipeline_run_all_detectors(self, sample_dataframe):
        """Test running all detectors in the pipeline."""
        detectors = [
            EuclideanDetector(radius=3.0),
            IsolationForestDetector(contamination=0.1)
        ]
        pipeline = OutlierDetectionPipeline(detectors)

        result_df = pipeline.run_all_detectors(sample_dataframe)

        # Should have columns for both detectors
        assert 'Outlier_Euclidean' in result_df.columns
        assert 'Outlier_IsolationForest' in result_df.columns

        # Original columns should be preserved
        assert 'Text' in result_df.columns
        assert 'Embeddings' in result_df.columns
        assert 'Label' in result_df.columns

    def test_empty_dataframe_handling(self):
        """Test pipeline behavior with empty DataFrame."""
        detectors = [EuclideanDetector()]
        pipeline = OutlierDetectionPipeline(detectors)

        empty_df = pd.DataFrame(columns=['Text', 'Embeddings', 'Label'])
        result_df = pipeline.run_all_detectors(empty_df)

        assert len(result_df) == 0
        assert 'Outlier_Euclidean' in result_df.columns
