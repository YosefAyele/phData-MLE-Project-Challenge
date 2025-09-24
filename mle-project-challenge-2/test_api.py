"""
Test script to demonstrate the FastAPI house price prediction service.
Uses examples from future_unseen_examples.csv to test the API endpoints.
"""
import json
import time
import requests
import pandas as pd
from typing import List, Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"
FUTURE_EXAMPLES_PATH = "data/future_unseen_examples.csv"


class HousePriceAPITester:
    """Test client for the House Price Prediction API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("🏥 Testing health check endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"   ✅ Health check passed: {health_data['status']}")
                print(f"   📊 Model loaded: {health_data['model_loaded']}")
                print(f"   📊 Demographics loaded: {health_data['demographics_loaded']}")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test the model info endpoint."""
        print("📊 Testing model info endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                model_info = response.json()
                print(f"   ✅ Model type: {model_info['model_type']}")
                print(f"   📈 Features count: {model_info['features_count']}")
                print(f"   🎯 R² Score: {model_info['performance_metrics'].get('r2', 'N/A'):.4f}")
                print(f"   💰 MAE: ${model_info['performance_metrics'].get('mae', 'N/A'):,.0f}")
                return True
            else:
                print(f"   ❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Model info error: {e}")
            return False
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from future_unseen_examples.csv."""
        print("📂 Loading test data...")
        try:
            df = pd.read_csv(FUTURE_EXAMPLES_PATH)
            
            # Convert DataFrame to list of dictionaries
            test_examples = df.head(10).to_dict('records')  # Use first 10 examples
            
            # Clean the data (handle any NaN values)
            for example in test_examples:
                for key, value in example.items():
                    if pd.isna(value):
                        if key in ['yr_renovated']:
                            example[key] = 0
                        elif key in ['bathrooms', 'floors']:
                            example[key] = float(example[key]) if not pd.isna(value) else 1.0
                        else:
                            example[key] = 0
                    elif key == 'zipcode':
                        example[key] = str(int(value)).zfill(5)  # Ensure 5-digit zipcode
            
            print(f"   ✅ Loaded {len(test_examples)} test examples")
            return test_examples
            
        except Exception as e:
            print(f"   ❌ Failed to load test data: {e}")
            return []
    
    def test_single_prediction(self, house_data: Dict[str, Any]) -> bool:
        """Test single house prediction."""
        print(f"🏠 Testing single prediction for zipcode {house_data['zipcode']}...")
        try:
            payload = {
                "house_features": house_data
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/predict", json=payload)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                predicted_price = result['predicted_price']
                metadata = result['prediction_metadata']
                
                print(f"   ✅ Predicted price: ${predicted_price:,.2f}")
                print(f"   ⏱️  Response time: {response_time:.1f}ms")
                print(f"   📊 Demographics found: {metadata['zipcode_demographics_found']}")
                print(f"   🏠 Features: {house_data['bedrooms']}BR/{house_data['bathrooms']}BA, {house_data['sqft_living']}sqft")
                return True
            else:
                print(f"   ❌ Prediction failed: {response.status_code}")
                print(f"   💬 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            return False
    
    def test_minimal_prediction(self, house_data: Dict[str, Any]) -> bool:
        """Test minimal house prediction."""
        print(f"🏠 Testing minimal prediction for zipcode {house_data['zipcode']}...")
        try:
            # Extract only minimal features
            minimal_features = {
                'sqft_living': house_data['sqft_living'],
                'bedrooms': house_data['bedrooms'],
                'bathrooms': house_data['bathrooms'],
                'sqft_above': house_data['sqft_above'],
                'floors': house_data['floors'],
                'sqft_lot': house_data['sqft_lot'],
                'sqft_basement': house_data['sqft_basement'],
                'zipcode': house_data['zipcode']
            }
            
            payload = {
                "house_features": minimal_features
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/predict/minimal", json=payload)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                predicted_price = result['predicted_price']
                
                print(f"   ✅ Minimal predicted price: ${predicted_price:,.2f}")
                print(f"   ⏱️  Response time: {response_time:.1f}ms")
                return True
            else:
                print(f"   ❌ Minimal prediction failed: {response.status_code}")
                print(f"   💬 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Minimal prediction error: {e}")
            return False
    
    def test_batch_prediction(self, houses_data: List[Dict[str, Any]]) -> bool:
        """Test batch house prediction."""
        print(f"🏘️  Testing batch prediction for {len(houses_data)} houses...")
        try:
            payload = {
                "houses": houses_data[:5]  # Test with first 5 houses
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                batch_metadata = result['batch_metadata']
                
                print(f"   ✅ Batch prediction successful")
                print(f"   🏠 Houses processed: {batch_metadata['batch_size']}")
                print(f"   ⏱️  Total time: {response_time:.1f}ms")
                print(f"   📊 Avg time per house: {batch_metadata['average_time_per_prediction_ms']:.1f}ms")
                
                # Show sample predictions
                for i, prediction in enumerate(predictions[:3]):
                    price = prediction['predicted_price']
                    print(f"   💰 House {i+1}: ${price:,.2f}")
                
                return True
            else:
                print(f"   ❌ Batch prediction failed: {response.status_code}")
                print(f"   💬 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Batch prediction error: {e}")
            return False
    
    def test_zipcodes_endpoint(self) -> bool:
        """Test the available zipcodes endpoint."""
        print("📫 Testing zipcodes endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/zipcodes")
            if response.status_code == 200:
                zipcodes = response.json()
                print(f"   ✅ Available zipcodes: {len(zipcodes)}")
                print(f"   📋 Sample zipcodes: {zipcodes[:5]}")
                return True
            else:
                print(f"   ❌ Zipcodes endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Zipcodes endpoint error: {e}")
            return False
    
    def run_comprehensive_test(self) -> None:
        """Run all tests comprehensively."""
        print("🚀 Starting comprehensive API testing...\n")
        
        # Track test results
        test_results = {}
        
        # Test 1: Health check
        test_results['health'] = self.test_health_check()
        print()
        
        # Test 2: Model info
        test_results['model_info'] = self.test_model_info()
        print()
        
        # Test 3: Available zipcodes
        test_results['zipcodes'] = self.test_zipcodes_endpoint()
        print()
        
        # Load test data
        test_data = self.load_test_data()
        if not test_data:
            print("❌ Cannot proceed without test data")
            return
        print()
        
        # Test 4: Single prediction
        test_results['single_prediction'] = self.test_single_prediction(test_data[0])
        print()
        
        # Test 5: Minimal prediction
        test_results['minimal_prediction'] = self.test_minimal_prediction(test_data[1])
        print()
        
        # Test 6: Batch prediction
        test_results['batch_prediction'] = self.test_batch_prediction(test_data)
        print()
        
        # Summary
        print("📋 Test Results Summary:")
        print("=" * 50)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the API service.")


def main():
    """Main function to run API tests."""
    print("🏠 House Price Prediction API Tester")
    print("=" * 50)
    
    # Instructions for running the API
    print("\n📋 Instructions:")
    print("1. Make sure the API server is running:")
    print("   cd /path/to/project")
    print("   python -m uvicorn app.main:app --reload")
    print("2. The API should be available at http://localhost:8000")
    print("3. API documentation at http://localhost:8000/docs")
    print("\n" + "=" * 50)
    
    # Create tester and run tests
    tester = HousePriceAPITester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()
