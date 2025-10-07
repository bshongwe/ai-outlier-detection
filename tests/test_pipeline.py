"""Tests for the main pipeline functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

def test_basic_imports():
    """Test that core modules can be imported."""
    from src.config import Config
    from src.outlier_detectors import EuclideanDetector
    assert Config is not None
    assert EuclideanDetector is not None

def test_config_creation():
    """Test configuration creation."""
    from src.config import Config
    config = Config()
    assert config.api.base_url == "https://api.studio.nebius.com/v1/"
    assert config.models.embedding == "BAAI/bge-en-icl"

def test_euclidean_detector():
    """Test Euclidean detector initialization."""
    from src.outlier_detectors import EuclideanDetector
    detector = EuclideanDetector(radius=0.5)
    assert detector.radius == 0.5
    assert detector.name == "Euclidean"

def test_isolation_forest_detector():
    """Test Isolation Forest detector initialization."""
    from src.outlier_detectors import IsolationForestDetector
    detector = IsolationForestDetector(contamination=0.1)
    assert detector.contamination == 0.1
    assert detector.name == "IsolationForest"

def test_data_preprocessing():
    """Test data preprocessing initialization."""
    from src.data_preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor(max_text_length=1000, sample_size=100)
    assert preprocessor.max_text_length == 1000
    assert preprocessor.sample_size == 100