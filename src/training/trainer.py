"""
Model training utilities for Vertex AI.
"""

import logging
from typing import Dict, Any, Optional, List
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomJob, CustomTrainingJob

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Utility class for training models on Vertex AI."""
    
    def __init__(self, 
                 project_id: str, 
                 region: str,
                 staging_bucket: str):
        """
        Initialize model trainer.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            staging_bucket: GCS bucket for staging artifacts
        """
        self.project_id = project_id
        self.region = region
        self.staging_bucket = staging_bucket
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=staging_bucket
        )
    
    def create_custom_training_job(self,
                                 display_name: str,
                                 script_path: str,
                                 container_uri: str,
                                 requirements: Optional[List[str]] = None,
                                 machine_type: str = "n1-standard-4",
                                 replica_count: int = 1,
                                 args: Optional[List[str]] = None) -> CustomTrainingJob:
        """
        Create a custom training job.
        
        Args:
            display_name: Display name for the training job
            script_path: Path to the training script
            container_uri: URI of the training container
            requirements: Python package requirements
            machine_type: Machine type for training
            replica_count: Number of replicas
            args: Additional arguments for the training script
            
        Returns:
            CustomTrainingJob object
        """
        try:
            job = CustomTrainingJob(
                display_name=display_name,
                script_path=script_path,
                container_uri=container_uri,
                requirements=requirements or [],
                model_serving_container_image_uri=container_uri,
            )
            
            logger.info(f"Created custom training job: {display_name}")
            return job
        except Exception as e:
            logger.error(f"Failed to create training job {display_name}: {str(e)}")
            raise
    
    def submit_training_job(self,
                          job: CustomTrainingJob,
                          args: Optional[Dict[str, Any]] = None,
                          environment_variables: Optional[Dict[str, str]] = None) -> aiplatform.Model:
        """
        Submit and run a training job.
        
        Args:
            job: CustomTrainingJob to submit
            args: Arguments for the training job
            environment_variables: Environment variables for the job
            
        Returns:
            Trained model object
        """
        try:
            model = job.run(
                args=args,
                environment_variables=environment_variables,
                sync=True
            )
            
            logger.info(f"Training job completed successfully. Model: {model.display_name}")
            return model
        except Exception as e:
            logger.error(f"Training job failed: {str(e)}")
            raise