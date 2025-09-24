"""
Simple script to run the FastAPI application.
"""
import uvicorn
from app.utils.config import HOST, PORT, LOG_LEVEL

if __name__ == "__main__":
    print("🚀 Starting House Price Prediction API...")
    print(f"📍 Server will be available at: http://{HOST}:{PORT}")
    print(f"📚 API Documentation: http://{HOST}:{PORT}/docs")
    print(f"🔧 Alternative docs: http://{HOST}:{PORT}/redoc")
    print("Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level=LOG_LEVEL
    )
