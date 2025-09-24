import logging
import time
from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from app.models.prediction import (
    PredictionRequest, PredictionRequestMinimal, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse, HealthCheckResponse,
    ModelInfoResponse, ErrorResponse
)
from app.services.model_service import model_service
from app.services.data_service import data_service
from app.utils.config import API_TITLE, API_VERSION

logger = logging.getLogger(__name__)

router = APIRouter()


async def check_service_health():
    if not model_service.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not healthy"
        )
    
    if not data_service.demographics_data is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data service is not healthy"
        )


@router.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    model_healthy = model_service.is_healthy()
    data_healthy = data_service.demographics_data is not None
    
    status_value = "healthy" if (model_healthy and data_healthy) else "unhealthy"
    
    return HealthCheckResponse(
        status=status_value,
        timestamp=datetime.now().isoformat(),
        model_loaded=model_healthy,
        demographics_loaded=data_healthy,
        version=API_VERSION
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    model_info = model_service.get_model_info()
    
    if "error" in model_info:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=model_info["error"]
        )
    
    return ModelInfoResponse(
        model_type=model_info["model_type"],
        model_version=model_info["model_version"],
        features_count=model_info["features_count"],
        feature_names=model_info["feature_names"],
        training_data_size=model_info.get("training_data_size", 0),
        performance_metrics=model_info.get("performance_metrics", {}),
        last_updated=model_info["last_updated"]
    )


@router.get("/model/features", response_model=List[str])
async def get_model_features():
    return model_service.model_features


@router.get("/zipcodes", response_model=List[str])
async def get_available_zipcodes():
    return data_service.get_available_zipcodes()


@router.get("/demographics/summary", response_model=Dict[str, Any])
async def get_demographics_summary():
    return data_service.get_demographics_summary()


@router.post("/predict", response_model=PredictionResponse)
async def predict_house_price(
    request: PredictionRequest,
    _: None = Depends(check_service_health)
):
    try:
        features_df = data_service.prepare_features_for_prediction(
            request.house_features,
            model_service.model_features
        )
        
        zipcode_found = data_service.validate_zipcode(request.house_features.zipcode)
        
        result = model_service.predict_with_metadata(
            features_df,
            zipcode_demographics_found=zipcode_found
        )
        
        return PredictionResponse(
            predicted_price=result["predicted_price"],
            prediction_metadata=result["prediction_metadata"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/minimal", response_model=PredictionResponse)
async def predict_house_price_minimal(
    request: PredictionRequestMinimal,
    _: None = Depends(check_service_health)
):
    try:
        features_df = data_service.prepare_minimal_features(
            request.house_features,
            model_service.model_features
        )
        
        zipcode_found = data_service.validate_zipcode(request.house_features.zipcode)
        
        result = model_service.predict_with_metadata(
            features_df,
            zipcode_demographics_found=zipcode_found
        )
        
        return PredictionResponse(
            predicted_price=result["predicted_price"],
            prediction_metadata=result["prediction_metadata"]
        )
        
    except Exception as e:
        logger.error(f"Minimal prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_house_prices(
    request: BatchPredictionRequest,
    _: None = Depends(check_service_health)
):
    try:
        start_time = time.time()
        predictions = []
        
        for house_features in request.houses:
            features_df = data_service.prepare_features_for_prediction(
                house_features,
                model_service.model_features
            )
            
            zipcode_found = data_service.validate_zipcode(house_features.zipcode)
            
            result = model_service.predict_with_metadata(
                features_df,
                zipcode_demographics_found=zipcode_found
            )
            
            predictions.append(PredictionResponse(
                predicted_price=result["predicted_price"],
                prediction_metadata=result["prediction_metadata"]
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        batch_metadata = {
            "batch_size": len(request.houses),
            "total_processing_time_ms": processing_time,
            "average_time_per_prediction_ms": processing_time / len(request.houses),
            "timestamp": datetime.now().isoformat()
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_metadata=batch_metadata
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_service_statistics():
    model_stats = model_service.get_prediction_statistics()
    demographics_summary = data_service.get_demographics_summary()
    
    return {
        "service": {
            "uptime_hours": model_stats["uptime_hours"],
            "total_predictions": model_stats["total_predictions"],
            "service_started": model_stats["model_loaded_at"]
        },
        "model": model_service.get_model_info(),
        "data": demographics_summary
    }
