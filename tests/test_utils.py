"""
Test cases for utility functions.
"""

import pytest
from unittest.mock import Mock, patch
from src.utils.config import Config
from src.utils.vertex_client import VertexAIClient

class TestConfig:
    """Test cases for configuration management."""
    
    def test_config_initialization(self, temp_config):
        """Test configuration initialization."""
        assert temp_config.project_id == "test-project"
        assert temp_config.region == "us-central1"
    
    def test_config_validation(self, temp_config):
        """Test configuration validation."""
        assert temp_config.validate() is True
    
    def test_config_validation_missing_project(self):
        """Test configuration validation with missing project ID."""
        config = Config()
        config.project_id = None
        
        with pytest.raises(ValueError, match="Missing required configuration"):
            config.validate()

class TestVertexAIClient:
    """Test cases for Vertex AI client."""
    
    @patch('src.utils.vertex_client.aiplatform')
    def test_client_initialization(self, mock_aiplatform):
        """Test client initialization."""
        client = VertexAIClient("test-project", "us-central1")
        
        mock_aiplatform.init.assert_called_once_with(
            project="test-project",
            location="us-central1",
            credentials="us-central1"
        )
    
    @patch('src.utils.vertex_client.aiplatform')
    def test_list_models(self, mock_aiplatform):
        """Test listing models."""
        # Setup mock
        mock_model = Mock()
        mock_aiplatform.Model.list.return_value = [mock_model]
        
        client = VertexAIClient("test-project", "us-central1")
        models = client.list_models()
        
        assert len(models) == 1
        mock_aiplatform.Model.list.assert_called_once()