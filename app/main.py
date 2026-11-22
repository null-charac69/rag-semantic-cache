"""
FastAPI Application
===================
Main entry point for the RAG API server.

Features:
- CORS enabled (for web frontends)
- Error handling
- Request logging
- Automatic API documentation (Swagger)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys
import time

from app.api.routes import router
from app.core.config import settings, ensure_directories, validate_config


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)

# File logging (writes to logs/rag_api.log; skip if not writable)
try:
    logger.add(
        "logs/rag_api.log",
        rotation="10 MB",
        retention="7 days",
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
    )
except (PermissionError, OSError):
    pass  # stdout only if logs/ is not writable


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RAG Semantic Cache API",
    description="Retrieval Augmented Generation with Semantic Caching",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc"     # ReDoc at /redoc
)


# ============================================================
# MIDDLEWARE
# ============================================================

# CORS - Allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests.
    
    Logs:
    - Method, path
    - Processing time
    - Status code
    """
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000  # ms
    
    # Log response
    logger.info(
        f"← {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.0f}ms"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response


# ============================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.
    
    - Validate configuration
    - Create directories
    - Initialize services
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting RAG Semantic Cache API")
    logger.info("=" * 60)
    
    try:
        # Ensure directories exist
        ensure_directories()
        
        # Validate configuration
        validate_config()
        
        # Warm up services (initialize singletons)
        from app.services.rag_pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        
        # Health check
        health = pipeline.health_check()
        logger.info(f"Health check: {health['status']}")
        
        logger.success("✅ API startup complete!")
        logger.info(f"📚 Swagger docs: http://localhost:8000/docs")
        logger.info(f"📖 ReDoc: http://localhost:8000/redoc")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("🛑 Shutting down RAG API...")
    logger.success("✅ Shutdown complete")


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler.
    
    Catches all unhandled exceptions and returns JSON response.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


# ============================================================
# ROUTES
# ============================================================

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG"])

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns API info and links.
    """
    return {
        "name": "RAG Semantic Cache API",
        "version": "1.0.0",
        "status": "running",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "query": "/api/v1/query",
            "upload": "/api/v1/upload",
            "health": "/api/v1/health",
            "stats": "/api/v1/stats"
        }
    }


# ============================================================
# MAIN (for development)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting development server...")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
