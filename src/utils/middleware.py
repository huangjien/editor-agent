"""Middleware components for the editor agent application."""

import time
from typing import Dict
from uuid import uuid4

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from src.config.settings import get_settings
from src.utils.logging import get_logger, RequestLogger

settings = get_settings()
logger = get_logger(__name__)
request_logger = RequestLogger()


class RequestIDMiddleware(BaseHTTPMiddleware):
  """Middleware to add unique request ID to each request."""

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    request_id = str(uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
  """Middleware to log HTTP requests and responses."""

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    start_time = time.time()

    # Log request
    request_logger.log_request(
      method=request.method, url=str(request.url), headers=dict(request.headers)
    )

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log response
    request_logger.log_response(
      status_code=response.status_code, response_time=duration
    )

    return response


class RateLimitMiddleware(BaseHTTPMiddleware):
  """Simple in-memory rate limiting middleware."""

  def __init__(self, app, requests_per_minute: int = None):
    super().__init__(app)
    self.requests_per_minute = (
      requests_per_minute or settings.rate_limit_requests_per_minute
    )
    self.request_counts: Dict[str, list] = {}
    self.cleanup_interval = 60  # seconds
    self.last_cleanup = time.time()

  def _get_client_id(self, request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Use API key if available, otherwise use IP address
    api_key = request.headers.get("X-API-Key")
    if api_key:
      return f"api_key:{api_key}"

    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"

  def _cleanup_old_requests(self):
    """Remove old request timestamps."""
    current_time = time.time()
    if current_time - self.last_cleanup < self.cleanup_interval:
      return

    cutoff_time = current_time - 60  # 1 minute ago
    for client_id in list(self.request_counts.keys()):
      self.request_counts[client_id] = [
        timestamp
        for timestamp in self.request_counts[client_id]
        if timestamp > cutoff_time
      ]
      if not self.request_counts[client_id]:
        del self.request_counts[client_id]

    self.last_cleanup = current_time

  def _is_rate_limited(self, client_id: str) -> bool:
    """Check if client is rate limited."""
    current_time = time.time()
    minute_ago = current_time - 60

    # Get recent requests for this client
    if client_id not in self.request_counts:
      self.request_counts[client_id] = []

    # Filter to requests in the last minute
    recent_requests = [
      timestamp
      for timestamp in self.request_counts[client_id]
      if timestamp > minute_ago
    ]
    self.request_counts[client_id] = recent_requests

    # Check if limit exceeded
    return len(recent_requests) >= self.requests_per_minute

  def _record_request(self, client_id: str):
    """Record a new request for the client."""
    current_time = time.time()
    if client_id not in self.request_counts:
      self.request_counts[client_id] = []
    self.request_counts[client_id].append(current_time)

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/"]:
      return await call_next(request)

    # Cleanup old requests periodically
    self._cleanup_old_requests()

    # Get client identifier
    client_id = self._get_client_id(request)

    # Check rate limit
    if self._is_rate_limited(client_id):
      logger.warning(
        f"Rate limit exceeded for client: {client_id}",
        extra={
          "client_id": client_id,
          "requests_per_minute": self.requests_per_minute,
          "request_url": str(request.url),
          "request_method": request.method,
        },
      )

      return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
          "error": {
            "code": "RATE_LIMIT_EXCEEDED",
            "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
            "details": {"limit": self.requests_per_minute, "window": "1 minute"},
          }
        },
        headers={
          "X-RateLimit-Limit": str(self.requests_per_minute),
          "X-RateLimit-Window": "60",
          "Retry-After": "60",
        },
      )

    # Record the request
    self._record_request(client_id)

    # Process request
    response = await call_next(request)

    # Add rate limit headers
    remaining_requests = max(
      0, self.requests_per_minute - len(self.request_counts.get(client_id, []))
    )
    response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(remaining_requests)
    response.headers["X-RateLimit-Window"] = "60"

    return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
  """Middleware for API key authentication."""

  def __init__(self, app, require_api_key: bool = None, api_keys: list = None):
    super().__init__(app)
    self.require_api_key = (
      require_api_key if require_api_key is not None else settings.require_api_key
    )
    # Use provided api_keys or fall back to settings
    keys_to_use = api_keys if api_keys is not None else settings.api_keys
    self.valid_api_keys = set(keys_to_use) if keys_to_use else set()

  def _is_public_endpoint(self, path: str) -> bool:
    """Check if endpoint is public and doesn't require authentication."""
    public_endpoints = ["/", "/health", "/docs", "/redoc", "/openapi.json"]
    return path in public_endpoints or path.startswith("/static")

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    # Skip authentication for public endpoints
    if self._is_public_endpoint(request.url.path):
      return await call_next(request)

    # Skip if API key authentication is not required
    if not self.require_api_key:
      return await call_next(request)

    # Check for API key
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

    if not api_key:
      logger.warning(
        "Missing API key",
        extra={
          "request_url": str(request.url),
          "request_method": request.method,
          "client_ip": request.client.host if request.client else "unknown",
        },
      )
      return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
          "error": {
            "code": "MISSING_API_KEY",
            "message": "API key is required. Provide it in X-API-Key header or api_key query parameter.",
            "details": {},
          }
        },
      )

    # Validate API key
    if self.valid_api_keys and api_key not in self.valid_api_keys:
      logger.warning(
        "Invalid API key",
        extra={
          "api_key_prefix": api_key[:8] + "..." if len(api_key) > 8 else api_key,
          "request_url": str(request.url),
          "request_method": request.method,
          "client_ip": request.client.host if request.client else "unknown",
        },
      )
      return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
          "error": {
            "code": "INVALID_API_KEY",
            "message": "Invalid API key provided.",
            "details": {},
          }
        },
      )

    # Store API key in request state for later use
    request.state.api_key = api_key

    return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
  """Middleware to add security headers to responses."""

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    # Remove server header for security
    if "server" in response.headers:
      del response.headers["server"]

    return response


class CORSMiddleware(BaseHTTPMiddleware):
  """Custom CORS middleware with more control."""

  def __init__(
    self,
    app,
    allowed_origins: list = None,
    allowed_methods: list = None,
    allowed_headers: list = None,
  ):
    super().__init__(app)
    self.allowed_origins = allowed_origins or settings.cors_origins
    self.allowed_methods = allowed_methods or [
      "GET",
      "POST",
      "PUT",
      "DELETE",
      "OPTIONS",
    ]
    self.allowed_headers = allowed_headers or ["*"]

  def _is_origin_allowed(self, origin: str) -> bool:
    """Check if origin is allowed."""
    if "*" in self.allowed_origins:
      return True
    return origin in self.allowed_origins

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    origin = request.headers.get("origin")

    # Handle preflight requests
    if request.method == "OPTIONS":
      response = Response()
      if origin and self._is_origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = ", ".join(
          self.allowed_methods
        )
        response.headers["Access-Control-Allow-Headers"] = ", ".join(
          self.allowed_headers
        )
        response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
      return response

    # Process normal request
    response = await call_next(request)

    # Add CORS headers
    if origin and self._is_origin_allowed(origin):
      response.headers["Access-Control-Allow-Origin"] = origin
      response.headers["Access-Control-Allow-Credentials"] = "true"

    return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
  """Middleware to limit request body size."""

  def __init__(self, app, max_size: int = None):
    super().__init__(app)
    self.max_size = max_size or settings.max_request_size

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    content_length = request.headers.get("content-length")

    if content_length and int(content_length) > self.max_size:
      logger.warning(
        f"Request body too large: {content_length} bytes (max: {self.max_size})",
        extra={
          "content_length": content_length,
          "max_size": self.max_size,
          "request_url": str(request.url),
          "request_method": request.method,
        },
      )
      return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={
          "error": {
            "code": "REQUEST_TOO_LARGE",
            "message": f"Request body too large. Maximum size is {self.max_size} bytes.",
            "details": {
              "max_size": self.max_size,
              "received_size": int(content_length),
            },
          }
        },
      )

    return await call_next(request)


class HealthCheckMiddleware(BaseHTTPMiddleware):
  """Middleware to handle health checks efficiently."""

  async def dispatch(
    self, request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    # Handle health check requests directly
    if request.url.path == "/health" and request.method == "GET":
      from src.utils.monitoring import get_health_checker

      try:
        health_checker = get_health_checker()
        health_data = await health_checker.run_all_checks()
        return JSONResponse(status_code=200, content=health_data)
      except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
          status_code=500,
          content={
            "error": {
              "message": "Health check failed",
              "details": str(e)
            }
          }
        )

    # Let other requests pass through
    return await call_next(request)
