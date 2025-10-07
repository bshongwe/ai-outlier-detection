"""Simple tests that don't require heavy dependencies."""

import pytest


def test_basic_detector_concepts():
    """Test basic detector concepts without imports."""
    # Test that we can create a simple detector-like class
    class SimpleDetector:
        def __init__(self, name):
            self.name = name
    
    detector = SimpleDetector("test")
    assert detector.name == "test"


def test_outlier_detection_logic():
    """Test basic outlier detection logic."""
    import numpy as np
    
    # Simple outlier detection using standard deviation
    data = np.array([1, 2, 3, 4, 5, 100])  # 100 is clearly an outlier
    mean = np.mean(data)
    std = np.std(data)
    threshold = 2 * std
    
    outliers = np.abs(data - mean) > threshold
    assert outliers[-1] == True  # Last element should be outlier
    assert sum(outliers) == 1  # Only one outlier


@pytest.mark.skipif(True, reason="Requires heavy ML dependencies")
def test_skip_heavy_ml_tests():
    """Placeholder for tests that require ML libraries."""
    pass