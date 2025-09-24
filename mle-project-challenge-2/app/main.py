import logging
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.utils.config import (
    API_TITLE, API_DESCRIPTION, API_VERSION, API_CONTACT,
    ALLOWED_ORIGINS, LOG_LEVEL, LOG_FORMAT, config
)
from app.models.prediction import ErrorResponse
from app.services.model_service import model_service
from app.services.data_service import data_service
from app.router import router

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="RuntimeError",
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    
    try:
        config.validate_paths()
        logger.info("Configuration validation passed")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    if not model_service.is_healthy():
        logger.error("Model service is not healthy on startup")
        raise RuntimeError("Model service failed to start")
    
    if data_service.demographics_data is None:
        logger.error("Data service is not healthy on startup")
        raise RuntimeError("Data service failed to start")
    
    logger.info("All services started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {API_TITLE}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=LOG_LEVEL
    )