"""Tests for the main pipeline functionality."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.pipeline import AIOutlierDetectionPipeline
from src.config import Config

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        api={'base_url': 'test', 'api_key': 'test', 'timeout': 30},
        models={'embedding': 'test-model', 'llm': 'test-llm'},
        processing={'batch_size': 2, 'max_text_length': 100, 'sample_size': 10},
        outlier_detection={'euclidean_radius': 0.5, 'contamination_rate': 0.1, 'lof_neighbors': 5, 'confidence_level': 0.95}
    )

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Text': ['This is a normal document', 'Another normal text', 'Completely different outlier content'],
        'Class Name': ['normal', 'normal', 'outlier'],
        'Label': [0, 0, 1]
    })

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [
        np.random.rand(384),
        np.random.rand(384),
        np.random.rand(384)
    ]

class TestAIOutlierDetectionPipeline:
    """Test cases for the main pipeline."""
    
    @patch('src.pipeline.load_config')
    @patch('src.pipeline.OpenAI')
    def test_pipeline_initialization(self, mock_openai, mock_load_config, mock_config):
        """Test pipeline initialization."""
        mock_load_config.return_value = mock_config
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        pipeline = AIOutlierDetectionPipeline()
        
        assert pipeline.config == mock_config
        assert pipeline.client == mock_client
        mock_openai.assert_called_once()
    
    @patch('src.pipeline.load_config')
    @patch('src.pipeline.OpenAI')
    def test_detect_outliers_in_text(self, mock_openai, mock_load_config, mock_config, sample_embeddings):
        """Test custom text outlier detection."""
        mock_load_config.return_value = mock_config
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        pipeline = AIOutlierDetectionPipeline()
        
        # Mock embedding generation
        pipeline.embedding_generator.add_embeddings_to_dataframe = Mock()
        pipeline.embedding_generator.add_embeddings_to_dataframe.return_value = pd.DataFrame({
            'Text': ['text1', 'text2'],
            'Class Name': ['custom', 'custom'],
            'Label': [0, 1],
            'Embeddings': [sample_embeddings[0], sample_embeddings[1]]
        })
        
        # Mock outlier detection
        pipeline.outlier_pipeline.run_all_detectors = Mock()
        pipeline.outlier_pipeline.run_all_detectors.return_value = pd.DataFrame({
            'Text': ['text1', 'text2'],
            'Class Name': ['custom', 'custom'],
            'Label': [0, 1],
            'Embeddings': [sample_embeddings[0], sample_embeddings[1]],
            'Outlier_IsolationForest': [False, True]
        })
        
        texts = ['text1', 'text2']
        results = pipeline.detect_outliers_in_text(texts)
        
        assert 'IsolationForest' in results
        assert results['IsolationForest']['outlier_count'] == 1
        assert results['IsolationForest']['outlier_indices'] == [1]

@pytest.mark.asyncio
class TestAPIIntegration:
    """Test API integration functionality."""
    
    def test_api_import(self):
        """Test that API module can be imported."""
        try:
            import api
            assert hasattr(api, 'app')
        except ImportError:
            pytest.skip("FastAPI not available")

class TestCLIIntegration:
    """Test CLI integration functionality."""
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        try:
            import cli
            assert hasattr(cli, 'cli')
        except ImportError:
            pytest.skip("Click not available")

class TestConfigManagement:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading from YAML."""
        from src.config import load_config, Config
        
        # Test with default values
        config = Config()
        assert config.api.base_url == "https://api.studio.nebius.com/v1/"
        assert config.models.embedding == "BAAI/bge-en-icl"
    
    def test_config_with_env_override(self):
        """Test configuration override with environment variables."""
        with patch.dict(os.environ, {'NEBIUS_API_KEY': 'test-key'}):
            from src.config import load_config
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("api:\n  base_url: 'test-url'\n")
                config_path = f.name
            
            try:
                config = load_config(config_path)
                assert config.api.api_key == 'test-key'
            finally:
                os.unlink(config_path)