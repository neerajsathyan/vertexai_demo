"""
Prediction utilities for Vertex AI models.
"""

import logging
from typing import List, Dict, Any, Optional
import json
import numpy as np
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Utility class for making predictions with Vertex AI models."""
    
    def __init__(self, 
                 project_id: str, 
                 region: str,
                 model_name: str,
                 endpoint_id: Optional[str] = None):
        """
        Initialize model predictor.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            model_name: Name of the deployed model
            endpoint_id: Optional endpoint ID for online predictions
        """
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.endpoint_id = endpoint_id
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=region
        )
        
        # Get endpoint if provided
        if endpoint_id:
            self.endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
        else:
            self.endpoint = None
    
    def predict_online(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make online predictions using a deployed endpoint.
        
        Args:
            instances: List of instances to predict
            
        Returns:
            List of prediction results
        """
        if not self.endpoint:
            raise ValueError("No endpoint configured for online predictions")
        
        try:
            predictions = self.endpoint.predict(instances=instances)
            logger.info(f"Made {len(instances)} online predictions")
            return predictions.predictions
        except Exception as e:
            logger.error(f"Online prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, 
                     input_uri: str, 
                     output_uri: str,
                     instances_format: str = "jsonl") -> aiplatform.BatchPredictionJob:
        """
        Create a batch prediction job.
        
        Args:
            input_uri: GCS URI of input data
            output_uri: GCS URI for output results
            instances_format: Format of input instances (jsonl, csv, etc.)
            
        Returns:
            BatchPredictionJob object
        """
        try:
            # Get model
            model = aiplatform.Model.list(filter=f'display_name="{self.model_name}"')[0]
            
            # Create batch prediction job
            job = model.batch_predict(
                job_display_name=f"batch-prediction-{self.model_name}",
                gcs_source=input_uri,
                gcs_destination_prefix=output_uri,
                instances_format=instances_format,
                sync=False
            )
            
            logger.info(f"Created batch prediction job: {job.display_name}")
            return job
        except Exception as e:
            logger.error(f"Batch prediction job creation failed: {str(e)}")
            raise
    
    def preprocess_instance(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess raw data into model input format.
        
        Args:
            raw_data: Raw input data
            
        Returns:
            Preprocessed instance
        """
        # This is a placeholder - implement your specific preprocessing logic
        # For example, feature scaling, encoding, etc.
        
        processed = {}
        for key, value in raw_data.items():
            if isinstance(value, (int, float)):
                processed[key] = float(value)
            else:
                processed[key] = str(value)
        
        return processed
    
    def postprocess_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess model prediction into desired output format.
        
        Args:
            prediction: Raw model prediction
            
        Returns:
            Processed prediction result
        """
        # This is a placeholder - implement your specific postprocessing logic
        # For example, confidence thresholding, class mapping, etc.
        
        return {
            'prediction': prediction.get('prediction', None),
            'confidence': prediction.get('confidence', None),
            'probabilities': prediction.get('probabilities', None)
        }