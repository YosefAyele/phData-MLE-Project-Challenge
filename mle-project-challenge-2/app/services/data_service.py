import logging
import pandas as pd
from typing import Dict, List, Optional, Set
from pathlib import Path

from app.utils.config import ZIPCODE_DEMOGRAPHICS_PATH
from app.models.prediction import HouseFeaturesBase, HouseFeaturesFull, HouseFeaturesMinimal

logger = logging.getLogger(__name__)


class DataService:    
    def __init__(self):
        self.demographics_data: Optional[pd.DataFrame] = None
        self.available_zipcodes: Set[str] = set()
        self._load_demographics()
    
    def _load_demographics(self) -> None:
        try:
            logger.info(f"Loading demographic data from {ZIPCODE_DEMOGRAPHICS_PATH}")
            self.demographics_data = pd.read_csv(
                ZIPCODE_DEMOGRAPHICS_PATH,
                dtype={'zipcode': str}
            )
            
            # Cache available zipcodes for quick lookup
            self.available_zipcodes = set(self.demographics_data['zipcode'].values)
            
            logger.info(f"Loaded demographic data for {len(self.available_zipcodes)} zipcodes")
            logger.info(f"Demographics features: {list(self.demographics_data.columns)}")
            
        except Exception as e:
            logger.error(f"Failed to load demographic data: {e}")
            raise
    
    def get_demographics_for_zipcode(self, zipcode: str) -> Optional[Dict]:
        if self.demographics_data is None:
            logger.error("Demographics data not loaded")
            return None
        
        zipcode_data = self.demographics_data[
            self.demographics_data['zipcode'] == zipcode
        ]
        
        if zipcode_data.empty:
            logger.warning(f"No demographic data found for zipcode: {zipcode}")
            return None
        
        # Convert to dictionary and remove zipcode column
        demo_dict = zipcode_data.iloc[0].to_dict()
        demo_dict.pop('zipcode', None)
        
        return demo_dict
    
    def prepare_features_for_prediction(
        self, 
        house_features: HouseFeaturesBase,
        model_features: List[str]
    ) -> pd.DataFrame:
     
        # Convert house features to dictionary
        house_dict = house_features.dict()
        zipcode = house_dict.pop('zipcode')
        
        # Get demographic data for the zipcode
        demographics = self.get_demographics_for_zipcode(zipcode)
        
        if demographics is None:
            # If no demographics found, fill with median values
            logger.warning(f"Using default demographics for zipcode {zipcode}")
            demographics = self._get_default_demographics()
        
        # Combine house features with demographics
        combined_features = {**house_dict, **demographics}
        
        # Create DataFrame with only the features needed by the model
        feature_data = {}
        for feature in model_features:
            if feature in combined_features:
                feature_data[feature] = combined_features[feature]
            else:
                logger.warning(f"Feature '{feature}' not found, using default value 0")
                feature_data[feature] = 0
        
        # Convert to DataFrame
        df = pd.DataFrame([feature_data])
        
        # Ensure correct column order matching model training
        df = df[model_features]
        
        return df
    
    def _get_default_demographics(self) -> Dict:
     
        if self.demographics_data is None:
            logger.error("Demographics data not loaded")
            return {}
        
        # Calculate median values for all demographic features
        demographic_columns = [
            col for col in self.demographics_data.columns 
            if col != 'zipcode'
        ]
        
        default_values = {}
        for col in demographic_columns:
            default_values[col] = self.demographics_data[col].median()
        
        return default_values
    
    def validate_zipcode(self, zipcode: str) -> bool:
    
        return zipcode in self.available_zipcodes
    
    def get_available_zipcodes(self) -> List[str]:
      
        return sorted(list(self.available_zipcodes))
    
    def get_demographics_summary(self) -> Dict:
     
        if self.demographics_data is None:
            return {}
        
        summary = {
            'total_zipcodes': len(self.available_zipcodes),
            'demographic_features': list(self.demographics_data.columns),
            'sample_zipcodes': sorted(list(self.available_zipcodes))[:10]
        }
        
        # Add some basic statistics
        numeric_columns = self.demographics_data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col != 'zipcode':
                summary[f'{col}_median'] = float(self.demographics_data[col].median())
                summary[f'{col}_mean'] = float(self.demographics_data[col].mean())
        
        return summary
    
    def prepare_minimal_features(
        self, 
        house_features: HouseFeaturesMinimal,
        model_features: List[str]
    ) -> pd.DataFrame:

        # Convert to full house features with defaults
        house_dict = house_features.dict()
        zipcode = house_dict['zipcode']
        
        # Get demographics for the zipcode
        demographics = self.get_demographics_for_zipcode(zipcode)
        if demographics is None:
            demographics = self._get_default_demographics()
        
        # Add default values for missing house features
        full_features = {
            **house_dict,
            **demographics,
            # Default values for features not in minimal set
            'waterfront': 0,
            'view': 0,
            'condition': 3,  # Average condition
            'grade': 7,      # Average grade
            'yr_built': 1975,  # Median year built
            'yr_renovated': 0,
            'lat': 47.5,     # Approximate Seattle latitude
            'long': -122.3,  # Approximate Seattle longitude
            'sqft_living15': house_dict['sqft_living'],  # Use same as house
            'sqft_lot15': house_dict['sqft_lot']         # Use same as house
        }
        
        # Remove zipcode from features
        full_features.pop('zipcode', None)
        
        # Create DataFrame with model features
        feature_data = {}
        for feature in model_features:
            if feature in full_features:
                feature_data[feature] = full_features[feature]
            else:
                logger.warning(f"Feature '{feature}' not found, using default value 0")
                feature_data[feature] = 0
        
        df = pd.DataFrame([feature_data])
        df = df[model_features]
        
        return df


# Global data service instance
data_service = DataService()
