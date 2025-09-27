"""
Test configuration and utilities.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from src.utils.config import Config

@pytest.fixture
def temp_config():
    """Fixture for temporary configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config()
        config.project_id = "test-project"
        config.region = "us-central1"
        config.staging_bucket = "gs://test-bucket"
        yield config

@pytest.fixture
def mock_vertex_ai():
    """Mock Vertex AI services."""
    with patch('google.cloud.aiplatform.init') as mock_init:
        yield mock_init