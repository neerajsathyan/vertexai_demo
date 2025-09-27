"""
Sample prediction script.
"""

import argparse
import json
import logging
from typing import Dict, Any, List

from src.prediction.predictor import ModelPredictor
from src.utils.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load test data for predictions.
    
    Args:
        data_path: Path to test data
        
    Returns:
        List of test instances
    """
    # For demo purposes, create sample data
    # Replace this with your actual data loading logic
    sample_data = [
        {f'feature_{i}': float(i * 0.1) for i in range(10)}
        for _ in range(5)
    ]
    
    logger.info(f"Loaded {len(sample_data)} test instances")
    return sample_data

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make predictions with Vertex AI model')
    parser.add_argument('--model-name', type=str, required=True,
                      help='Name of the deployed model')
    parser.add_argument('--endpoint-id', type=str,
                      help='Endpoint ID for online predictions')
    parser.add_argument('--data-path', type=str, default='data/test.json',
                      help='Path to test data')
    parser.add_argument('--output-path', type=str, default='predictions.json',
                      help='Path to save predictions')
    parser.add_argument('--batch', action='store_true',
                      help='Use batch prediction instead of online')
    
    args = parser.parse_args()
    
    # Validate configuration
    config.validate()
    
    # Initialize predictor
    predictor = ModelPredictor(
        project_id=config.project_id,
        region=config.region,
        model_name=args.model_name,
        endpoint_id=args.endpoint_id
    )
    
    if args.batch:
        # Batch prediction
        logger.info("Creating batch prediction job...")
        job = predictor.predict_batch(
            input_uri=args.data_path,
            output_uri=args.output_path
        )
        logger.info(f"Batch prediction job created: {job.display_name}")
        logger.info("Monitor the job in the Google Cloud Console")
    else:
        # Online prediction
        logger.info("Making online predictions...")
        
        # Load test data
        test_data = load_test_data(args.data_path)
        
        # Preprocess instances
        instances = [predictor.preprocess_instance(data) for data in test_data]
        
        # Make predictions
        predictions = predictor.predict_online(instances)
        
        # Postprocess predictions
        results = [predictor.postprocess_prediction(pred) for pred in predictions]
        
        # Save results
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Predictions saved to {args.output_path}")
        logger.info(f"Made {len(results)} predictions")

if __name__ == '__main__':
    main()