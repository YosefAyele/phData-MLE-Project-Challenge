import json
import pickle
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from app.utils.config import MODEL_PATH, MODEL_FEATURES_PATH, MODEL_VERSION, MODEL_TYPE
from app.models.prediction import PredictionMetadata

logger = logging.getLogger(__name__)


class ModelService:

    def __init__(self):
        self.model = None
        self.model_features: List[str] = []
        self.model_loaded_at: Optional[datetime] = None
        self.prediction_count: int = 0
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            
            # Load the pickled model
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load the feature list
            with open(MODEL_FEATURES_PATH, 'r') as f:
                self.model_features = json.load(f)
            
            self.model_loaded_at = datetime.now()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model type: {type(self.model)}")
            logger.info(f"Features count: {len(self.model_features)}")
            logger.info(f"Features: {self.model_features}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, features_df: pd.DataFrame) -> float:
    
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if features_df.empty:
            raise ValueError("Empty features DataFrame")
        
        # Ensure features are in the correct order
        if list(features_df.columns) != self.model_features:
            logger.warning("Feature order mismatch, reordering...")
            features_df = features_df[self.model_features]
        
        # Make prediction
        start_time = time.time()
        prediction = self.model.predict(features_df)
        prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Increment prediction counter
        self.prediction_count += 1
        
        # Log prediction details
        logger.info(f"Prediction made in {prediction_time:.2f}ms: ${prediction[0]:,.2f}")
        
        return float(prediction[0])
    
    def predict_with_metadata(
        self, 
        features_df: pd.DataFrame,
        zipcode_demographics_found: bool = True
    ) -> Dict[str, Any]:

        start_time = time.time()
        
        # Make prediction
        predicted_price = self.predict(features_df)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create metadata
        metadata = PredictionMetadata(
            model_version=MODEL_VERSION,
            prediction_timestamp=datetime.now().isoformat(),
            features_used=self.model_features.copy(),
            zipcode_demographics_found=zipcode_demographics_found,
            processing_time_ms=processing_time
        )
        
        return {
            'predicted_price': predicted_price,
            'prediction_metadata': metadata
        }
    
    def predict_batch(self, features_list: List[pd.DataFrame]) -> List[float]:

        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        predictions = []
        for features_df in features_list:
            prediction = self.predict(features_df)
            predictions.append(prediction)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:

        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Get model parameters
        model_params = {}
        if hasattr(self.model, 'named_steps'):
            # It's a pipeline
            for step_name, step in self.model.named_steps.items():
                if hasattr(step, 'get_params'):
                    model_params[step_name] = step.get_params()
        else:
            # Direct model
            if hasattr(self.model, 'get_params'):
                model_params = self.model.get_params()
        
        # Load performance metrics from evaluation report if available
        performance_metrics = {}
        try:
            eval_report_path = Path("model_evaluation_report.json")
            if eval_report_path.exists():
                with open(eval_report_path, 'r') as f:
                    eval_data = json.load(f)
                    performance_metrics = eval_data.get('test_performance', {})
        except Exception as e:
            logger.warning(f"Could not load evaluation report: {e}")
        
        return {
            'model_type': MODEL_TYPE,
            'model_version': MODEL_VERSION,
            'features_count': len(self.model_features),
            'feature_names': self.model_features.copy(),
            'model_loaded_at': self.model_loaded_at.isoformat() if self.model_loaded_at else None,
            'prediction_count': self.prediction_count,
            'model_parameters': model_params,
            'performance_metrics': performance_metrics,
            'last_updated': self.model_loaded_at.isoformat() if self.model_loaded_at else None
        }
    
    def is_healthy(self) -> bool:

        try:
            if self.model is None or not self.model_features:
                return False
            
            # Test prediction with dummy data
            dummy_features = pd.DataFrame([
                {feature: 0 for feature in self.model_features}
            ])
            # Set some reasonable values for key features
            dummy_features['sqft_living'] = 2000
            dummy_features['bedrooms'] = 3
            dummy_features['bathrooms'] = 2.0
            
            prediction = self.model.predict(dummy_features)
            
            # Check if prediction is reasonable (between $50k and $10M)
            return 50000 <= prediction[0] <= 10000000
            
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:

        # For KNN, we don't have built-in feature importance
        # Return empty dict or load from evaluation report
        try:
            eval_report_path = Path("model_evaluation_report.json")
            if eval_report_path.exists():
                with open(eval_report_path, 'r') as f:
                    eval_data = json.load(f)
                    feature_importance = eval_data.get('feature_importance', [])
                    
                    # Convert to dictionary
                    importance_dict = {}
                    for item in feature_importance:
                        importance_dict[item['feature']] = item['importance_mean']
                    
                    return importance_dict
        except Exception as e:
            logger.warning(f"Could not load feature importance: {e}")
        
        return {}
    
    def reload_model(self) -> bool:
 
        try:
            logger.info("Reloading model...")
            self._load_model()
            logger.info("Model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return False
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        return {
            'total_predictions': self.prediction_count,
            'model_loaded_at': self.model_loaded_at.isoformat() if self.model_loaded_at else None,
            'uptime_hours': (
                (datetime.now() - self.model_loaded_at).total_seconds() / 3600
                if self.model_loaded_at else 0
            )
        }


# Global model service instance
model_service = ModelService()
