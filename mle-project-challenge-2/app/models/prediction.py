from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import pandas as pd


class HouseFeaturesBase(BaseModel):
    
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_living: int = Field(..., ge=100, le=15000, description="Square footage of living space")
    sqft_lot: int = Field(..., ge=500, le=1000000, description="Square footage of lot")
    floors: float = Field(..., ge=1, le=5, description="Number of floors")
    sqft_above: int = Field(..., ge=100, le=15000, description="Square footage above ground")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Square footage of basement")
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit ZIP code")
    
    @validator('zipcode')
    def validate_zipcode(cls, v):
        if not v.isdigit():
            raise ValueError('Zipcode must contain only digits')
        return v
    
    @validator('sqft_basement')
    def validate_basement_size(cls, v, values):
        if 'sqft_living' in values and v > values['sqft_living']:
            raise ValueError('Basement square footage cannot exceed living space')
        return v


class HouseFeaturesFull(HouseFeaturesBase):
    
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront property (0 or 1)")
    view: int = Field(..., ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Property condition (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Building grade (1-13)")
    yr_built: int = Field(..., ge=1900, le=2025, description="Year built")
    yr_renovated: int = Field(..., ge=0, le=2025, description="Year renovated (0 if never)")
    lat: float = Field(..., ge=47.0, le=48.0, description="Latitude")
    long: float = Field(..., ge=-123.0, le=-121.0, description="Longitude")
    sqft_living15: int = Field(..., ge=100, le=15000, description="Living space of 15 nearest neighbors")
    sqft_lot15: int = Field(..., ge=500, le=1000000, description="Lot size of 15 nearest neighbors")
    
    @validator('yr_renovated')
    def validate_renovation_year(cls, v, values):
        """Validate that renovation year is after build year if not 0."""
        if v > 0 and 'yr_built' in values and v < values['yr_built']:
            raise ValueError('Renovation year cannot be before build year')
        return v


class HouseFeaturesMinimal(BaseModel):
    
    sqft_living: int = Field(..., ge=100, le=15000, description="Square footage of living space")
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms")
    sqft_above: int = Field(..., ge=100, le=15000, description="Square footage above ground")
    floors: float = Field(..., ge=1, le=5, description="Number of floors")
    sqft_lot: int = Field(..., ge=500, le=1000000, description="Square footage of lot")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Square footage of basement")
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit ZIP code")
    
    @validator('zipcode')
    def validate_zipcode(cls, v):
        if not v.isdigit():
            raise ValueError('Zipcode must contain only digits')
        return v
    
    @validator('sqft_basement')
    def validate_basement_size(cls, v, values):
        if 'sqft_living' in values and v > values['sqft_living']:
            raise ValueError('Basement square footage cannot exceed living space')
        return v


class PredictionRequest(BaseModel):    
    house_features: HouseFeaturesFull = Field(..., description="House features for prediction")


class PredictionRequestMinimal(BaseModel):
    
    house_features: HouseFeaturesMinimal = Field(..., description="Essential house features for prediction")


class BatchPredictionRequest(BaseModel):
    
    houses: List[HouseFeaturesFull] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="List of houses for batch prediction"
    )


class PredictionMetadata(BaseModel):
    
    model_version: str = Field(..., description="Version of the model used")
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")
    features_used: List[str] = Field(..., description="List of features used for prediction")
    zipcode_demographics_found: bool = Field(..., description="Whether demographic data was found for zipcode")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class PredictionResponse(BaseModel):
    
    predicted_price: float = Field(..., description="Predicted house price in USD")
    prediction_metadata: PredictionMetadata = Field(..., description="Prediction metadata")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Prediction confidence interval")


class BatchPredictionResponse(BaseModel):
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_metadata: Dict[str, Any] = Field(..., description="Batch processing metadata")


class HealthCheckResponse(BaseModel):
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    demographics_loaded: bool = Field(..., description="Whether demographic data is loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    
    model_type: str = Field(..., description="Type of machine learning model")
    model_version: str = Field(..., description="Version of the model")
    features_count: int = Field(..., description="Number of features used by the model")
    feature_names: List[str] = Field(..., description="Names of all features")
    training_data_size: int = Field(..., description="Size of training dataset")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    last_updated: str = Field(..., description="Last model update timestamp")


class ErrorResponse(BaseModel):
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


# Utility functions for model conversion
def house_features_to_dataframe(house_features: HouseFeaturesBase) -> pd.DataFrame:
    return pd.DataFrame([house_features.dict()])


def validate_zipcode_exists(zipcode: str, available_zipcodes: set) -> bool:
    return zipcode in available_zipcodes
