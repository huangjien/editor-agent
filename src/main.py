"""Main FastAPI application entry point for the editor agent."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from src.api.routes import router
from src.config.settings import get_settings
from src.utils.exceptions import (
  EditorAgentException,
  editor_agent_exception_handler,
  http_exception_handler,
  general_exception_handler,
)
from src.utils.middleware import (
  RequestIDMiddleware,
  RequestLoggingMiddleware,
  RateLimitMiddleware,
  APIKeyAuthMiddleware,
  SecurityHeadersMiddleware,
  RequestSizeLimitMiddleware,
  HealthCheckMiddleware,
)
from src.utils.logging import setup_logging, get_logger
from src.utils.monitoring import (
  get_health_checker,
  get_metrics_collector,
  get_system_monitor,
)


def create_app(settings_override=None):
  """Create FastAPI app with optional settings override."""
  settings = settings_override or get_settings()

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    yield

    # Shutdown
    logger.info("Shutting down application")

  # Create FastAPI app
  app = FastAPI(
    title=settings.app_name,
    description="AI Agent powered by LangGraph for code editing and assistance",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
  )

  # Add exception handlers
  app.add_exception_handler(EditorAgentException, editor_agent_exception_handler)
  app.add_exception_handler(HTTPException, http_exception_handler)
  app.add_exception_handler(Exception, general_exception_handler)

  # Add middleware (order matters - last added is executed first)
  app.add_middleware(SecurityHeadersMiddleware)
  app.add_middleware(RequestSizeLimitMiddleware, max_size=settings.max_file_size)
  app.add_middleware(HealthCheckMiddleware)
  app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_requests_per_minute if hasattr(settings, 'rate_limit_requests_per_minute') else None)
  app.add_middleware(APIKeyAuthMiddleware, require_api_key=settings.require_api_key, api_keys=settings.api_keys)
  app.add_middleware(RequestLoggingMiddleware)
  app.add_middleware(RequestIDMiddleware)

  # Add CORS middleware
  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  # Add trusted host middleware
  app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts,
  )

  # Include routers
  app.include_router(router, prefix="/api/v1")

  # Add route handlers
  @app.get("/")
  async def root():
    """Root endpoint."""
    from datetime import datetime, UTC

    return {
      "message": "Editor Agent API",
      "version": settings.app_version,
      "timestamp": datetime.now(UTC).isoformat(),
      "status": "running",
      "docs_url": "/docs" if settings.debug else None,
    }

  @app.get("/health")
  async def health_check():
    """Health check endpoint."""
    health_checker = get_health_checker()
    return await health_checker.run_all_checks()

  @app.get("/metrics")
  async def metrics():
    """Metrics endpoint."""
    metrics_collector = get_metrics_collector()
    return metrics_collector.get_metrics()

  @app.get("/system")
  async def system_info():
    """System information endpoint."""
    system_monitor = get_system_monitor()
    return system_monitor.get_system_info()

  return app


# Create the default app
app = create_app()

# Get settings for uvicorn
settings = get_settings()


if __name__ == "__main__":
  uvicorn.run(
    "src.main:app",
    host=settings.host,
    port=settings.port,
    reload=settings.debug,
    log_level=settings.log_level.lower(),
    access_log=True,
  )