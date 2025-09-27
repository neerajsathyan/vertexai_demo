"""
Configuration management for Vertex AI project.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Vertex AI project."""
    
    def __init__(self):
        # Google Cloud configuration
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.region = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
        self.credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Vertex AI configuration
        self.staging_bucket = os.getenv('VERTEX_AI_STAGING_BUCKET')
        self.model_display_name = os.getenv('MODEL_DISPLAY_NAME', 'custom-model')
        
        # Training configuration
        self.machine_type = os.getenv('TRAINING_MACHINE_TYPE', 'n1-standard-4')
        self.replica_count = int(os.getenv('TRAINING_REPLICA_COUNT', '1'))
        
        # Data configuration
        self.dataset_display_name = os.getenv('DATASET_DISPLAY_NAME', 'custom-dataset')
        
    def validate(self) -> bool:
        """Validate required configuration parameters."""
        required_configs = [
            ('project_id', self.project_id),
            ('region', self.region),
        ]
        
        missing_configs = [name for name, value in required_configs if not value]
        
        if missing_configs:
            raise ValueError(f"Missing required configuration: {', '.join(missing_configs)}")
        
        return True

# Global configuration instance
config = Config()