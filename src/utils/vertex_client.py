"""
Vertex AI client utilities and helper functions.
"""

from typing import Optional, Dict, Any, List
from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
import logging

logger = logging.getLogger(__name__)

class VertexAIClient:
    """Client for interacting with Google Cloud Vertex AI."""
    
    def __init__(self, project_id: str, region: str, credentials_path: Optional[str] = None):
        """
        Initialize Vertex AI client.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            credentials_path: Optional path to service account credentials
        """
        self.project_id = project_id
        self.region = region
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=region,
            credentials=credentials_path
        )
        
        logger.info(f"Initialized Vertex AI client for project {project_id} in region {region}")
    
    def create_dataset(self, 
                      display_name: str, 
                      metadata_schema_uri: str,
                      source_uris: Optional[List[str]] = None) -> aiplatform.Dataset:
        """
        Create a new dataset in Vertex AI.
        
        Args:
            display_name: Display name for the dataset
            metadata_schema_uri: URI for the dataset schema
            source_uris: Optional list of data source URIs
            
        Returns:
            Created dataset object
        """
        try:
            dataset = aiplatform.Dataset.create(
                display_name=display_name,
                metadata_schema_uri=metadata_schema_uri,
                source_uris=source_uris
            )
            logger.info(f"Created dataset: {display_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to create dataset {display_name}: {str(e)}")
            raise
    
    def list_models(self) -> List[aiplatform.Model]:
        """
        List all models in the project.
        
        Returns:
            List of model objects
        """
        try:
            models = aiplatform.Model.list()
            logger.info(f"Found {len(models)} models")
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise
    
    def get_model(self, model_id: str) -> aiplatform.Model:
        """
        Get a specific model by ID.
        
        Args:
            model_id: Model resource ID
            
        Returns:
            Model object
        """
        try:
            model = aiplatform.Model(model_name=model_id)
            logger.info(f"Retrieved model: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {str(e)}")
            raise