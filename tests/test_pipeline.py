"""Tests for the main pipeline functionality."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch


def test_config_creation():
    """Test configuration creation."""
    from src.config import Config
    config = Config()
    assert config.api.base_url == "https://api.studio.nebius.com/v1/"
    assert config.models.embedding == "BAAI/bge-en-icl"


@pytest.mark.skipif(True, reason="Requires heavy ML dependencies")
def test_ml_components():
    """Placeholder for ML component tests."""
    pass
