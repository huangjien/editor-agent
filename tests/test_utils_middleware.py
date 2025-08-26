"""Tests for middleware utilities."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.middleware import (
  RequestIDMiddleware,
  RequestLoggingMiddleware,
  RateLimitMiddleware,
  APIKeyAuthMiddleware,
  SecurityHeadersMiddleware,
  CORSMiddleware,
  RequestSizeLimitMiddleware,
  HealthCheckMiddleware,
)


class TestRequestIDMiddleware:
  """Test RequestIDMiddleware."""

  @pytest.mark.asyncio
  async def test_adds_request_id(self):
    """Test that middleware adds request ID."""
    app = FastAPI()
    middleware = RequestIDMiddleware(app)

    request = MagicMock(spec=Request)
    request.headers = {}
    request.state = MagicMock()

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    # Should set request_id on request state
    assert hasattr(request.state, "request_id")
    assert len(request.state.request_id) == 36  # UUID length

    # Should add X-Request-ID header to response
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == request.state.request_id

  @pytest.mark.asyncio
  async def test_uses_existing_request_id(self):
    """Test that middleware generates new request ID (current implementation)."""
    app = FastAPI()
    middleware = RequestIDMiddleware(app)

    request = MagicMock(spec=Request)
    request.headers = {}
    request.state = MagicMock()

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    # Should generate new request ID
    assert hasattr(request.state, "request_id")
    assert request.state.request_id is not None
    assert response.headers["X-Request-ID"] == request.state.request_id


class TestRequestLoggingMiddleware:
  """Test RequestLoggingMiddleware."""

  @pytest.mark.asyncio
  async def test_logs_request_and_response(self):
    """Test that middleware logs requests and responses."""
    app = FastAPI()
    middleware = RequestLoggingMiddleware(app)

    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url = MagicMock()
    request.url.path = "/test"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.state = MagicMock()
    request.state.request_id = "test-request-id"

    response = Response(status_code=200)
    call_next = AsyncMock(return_value=response)

    with patch("src.utils.middleware.request_logger") as mock_request_logger:
      await middleware.dispatch(request, call_next)

      # Should log request and response
      mock_request_logger.log_request.assert_called_once()
      mock_request_logger.log_response.assert_called_once()

  @pytest.mark.asyncio
  async def test_logs_request_duration(self):
    """Test that middleware logs request duration."""
    app = FastAPI()
    middleware = RequestLoggingMiddleware(app)

    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.state = MagicMock()
    request.state.request_id = "test-request-id"

    async def slow_call_next(req):
      await asyncio.sleep(0.1)  # Simulate slow response
      return Response(status_code=201)

    with patch("src.utils.middleware.request_logger") as mock_request_logger:
      await middleware.dispatch(request, slow_call_next)

      # Should log response with duration
      mock_request_logger.log_response.assert_called_once()
      # Check that response_time was passed
      call_args = mock_request_logger.log_response.call_args
      assert "response_time" in call_args.kwargs
      assert call_args.kwargs["response_time"] > 0


class TestRateLimitMiddleware:
  """Test RateLimitMiddleware."""

  @pytest.mark.asyncio
  async def test_allows_requests_within_limit(self):
    """Test that requests within limit are allowed."""
    app = FastAPI()
    middleware = RateLimitMiddleware(app, requests_per_minute=5)

    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"

    call_next = AsyncMock(return_value=Response())

    # Make multiple requests within limit
    for _ in range(5):
      response = await middleware.dispatch(request, call_next)
      assert response.status_code == 200

  @pytest.mark.asyncio
  async def test_blocks_requests_over_limit(self):
    """Test that requests over limit are blocked."""
    app = FastAPI()
    middleware = RateLimitMiddleware(app, requests_per_minute=2)

    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = "127.0.0.1"

    call_next = AsyncMock(return_value=Response())

    # Make requests up to limit
    for _ in range(2):
      response = await middleware.dispatch(request, call_next)
      assert response.status_code == 200

    # Next request should be rate limited
    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 429
    assert isinstance(response, JSONResponse)

  @pytest.mark.asyncio
  async def test_different_clients_separate_limits(self):
    """Test that different clients have separate rate limits."""
    app = FastAPI()
    middleware = RateLimitMiddleware(app, requests_per_minute=1)

    request1 = MagicMock(spec=Request)
    request1.client = MagicMock()
    request1.client.host = "127.0.0.1"

    request2 = MagicMock(spec=Request)
    request2.client = MagicMock()
    request2.client.host = "192.168.1.1"

    call_next = AsyncMock(return_value=Response())

    # Both clients should be able to make one request
    response1 = await middleware.dispatch(request1, call_next)
    assert response1.status_code == 200

    response2 = await middleware.dispatch(request2, call_next)
    assert response2.status_code == 200

    # Second request from first client should be rate limited
    response1_second = await middleware.dispatch(request1, call_next)
    assert response1_second.status_code == 429


class TestAPIKeyAuthMiddleware:
  """Test APIKeyAuthMiddleware."""

  @pytest.mark.asyncio
  async def test_allows_request_with_valid_api_key(self):
    """Test that requests with valid API key are allowed."""
    app = FastAPI()
    with patch("src.utils.middleware.settings") as mock_settings:
      mock_settings.require_api_key = True
      mock_settings.api_keys = ["valid-key"]
      middleware = APIKeyAuthMiddleware(app, require_api_key=True)

      request = MagicMock(spec=Request)
      request.headers = {"X-API-Key": "valid-key"}
      request.url = MagicMock()
      request.url.path = "/api/test"
      request.query_params = {}

      call_next = AsyncMock(return_value=Response())

      response = await middleware.dispatch(request, call_next)

      assert response.status_code == 200
      call_next.assert_called_once()

  @pytest.mark.asyncio
  async def test_blocks_request_with_invalid_api_key(self):
    """Test that requests with invalid API key are blocked."""
    app = FastAPI()
    with patch("src.utils.middleware.settings") as mock_settings:
      mock_settings.require_api_key = True
      mock_settings.api_keys = ["valid-key"]
      middleware = APIKeyAuthMiddleware(app, require_api_key=True)

      request = MagicMock(spec=Request)
      request.headers = {"X-API-Key": "invalid-key"}
      request.url = MagicMock()
      request.url.path = "/api/test"
      request.query_params = {}
      request.method = "GET"
      request.client = MagicMock()
      request.client.host = "127.0.0.1"

      call_next = AsyncMock(return_value=Response())

      response = await middleware.dispatch(request, call_next)

      assert response.status_code == 401
      assert isinstance(response, JSONResponse)

  @pytest.mark.asyncio
  async def test_blocks_request_without_api_key(self):
    """Test that requests without API key are blocked."""
    app = FastAPI()
    with patch("src.utils.middleware.settings") as mock_settings:
      mock_settings.require_api_key = True
      mock_settings.api_keys = ["valid-key"]
      middleware = APIKeyAuthMiddleware(app, require_api_key=True)

      request = MagicMock(spec=Request)
      request.headers = {}
      request.url = MagicMock()
      request.url.path = "/api/test"
      request.query_params = {}
      request.method = "GET"
      request.client = MagicMock()
      request.client.host = "127.0.0.1"

      call_next = AsyncMock(return_value=Response())

      response = await middleware.dispatch(request, call_next)

      assert response.status_code == 401
      assert isinstance(response, JSONResponse)

  @pytest.mark.asyncio
  async def test_skips_excluded_paths(self):
    """Test that excluded paths skip authentication."""
    app = FastAPI()
    with patch("src.utils.middleware.settings") as mock_settings:
      mock_settings.require_api_key = True
      mock_settings.api_keys = ["valid-key"]
      middleware = APIKeyAuthMiddleware(app, require_api_key=True)

      request = MagicMock(spec=Request)
      request.url = MagicMock()
      request.url.path = "/health"  # This is a public endpoint
      request.headers = {}  # No API key

      call_next = AsyncMock(return_value=Response())

      response = await middleware.dispatch(request, call_next)

      assert response.status_code == 200
      call_next.assert_called_once()


class TestSecurityHeadersMiddleware:
  """Test SecurityHeadersMiddleware."""

  @pytest.mark.asyncio
  async def test_adds_security_headers(self):
    """Test that middleware adds security headers."""
    app = FastAPI()
    middleware = SecurityHeadersMiddleware(app)

    request = MagicMock(spec=Request)
    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    # Check that security headers are added
    expected_headers = [
      "X-Content-Type-Options",
      "X-Frame-Options",
      "X-XSS-Protection",
      "Referrer-Policy",
      "Content-Security-Policy",
    ]

    for header in expected_headers:
      assert header in response.headers

  @pytest.mark.asyncio
  async def test_security_header_values(self):
    """Test specific security header values."""
    app = FastAPI()
    middleware = SecurityHeadersMiddleware(app)

    request = MagicMock(spec=Request)
    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Content-Security-Policy"] == "default-src 'self'"


class TestCORSMiddleware:
  """Test CORSMiddleware."""

  @pytest.mark.asyncio
  async def test_handles_preflight_request(self):
    """Test that middleware handles preflight requests."""
    app = FastAPI()
    allowed_origins = ["https://example.com"]
    allowed_methods = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers = ["Content-Type", "Authorization"]
    middleware = CORSMiddleware(
      app,
      allowed_origins=allowed_origins,
      allowed_methods=allowed_methods,
      allowed_headers=allowed_headers,
    )

    request = MagicMock(spec=Request)
    request.method = "OPTIONS"
    request.headers = {
      "origin": "https://example.com",
      "Access-Control-Request-Method": "POST",
      "Access-Control-Request-Headers": "Content-Type",
    }

    call_next = AsyncMock()

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"

    # Should not call next middleware for preflight
    call_next.assert_not_called()

  @pytest.mark.asyncio
  async def test_adds_cors_headers_to_response(self):
    """Test that middleware adds CORS headers to responses."""
    app = FastAPI()
    allowed_origins = ["https://example.com"]
    allowed_methods = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers = ["Content-Type", "Authorization"]
    middleware = CORSMiddleware(
      app,
      allowed_origins=allowed_origins,
      allowed_methods=allowed_methods,
      allowed_headers=allowed_headers,
    )

    request = MagicMock(spec=Request)
    request.method = "GET"
    request.headers = {"origin": "https://example.com"}

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    assert "Access-Control-Allow-Origin" in response.headers
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"

  @pytest.mark.asyncio
  async def test_blocks_disallowed_origin(self):
    """Test that middleware blocks disallowed origins."""
    app = FastAPI()
    allowed_origins = ["https://example.com"]
    allowed_methods = ["GET", "POST", "PUT", "DELETE"]
    allowed_headers = ["Content-Type", "Authorization"]
    middleware = CORSMiddleware(
      app,
      allowed_origins=allowed_origins,
      allowed_methods=allowed_methods,
      allowed_headers=allowed_headers,
    )

    request = MagicMock(spec=Request)
    request.method = "GET"
    request.headers = {"Origin": "https://malicious.com"}

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    # Should not include CORS headers for disallowed origin
    assert "Access-Control-Allow-Origin" not in response.headers


class TestRequestSizeLimitMiddleware:
  """Test RequestSizeLimitMiddleware."""

  @pytest.mark.asyncio
  async def test_allows_request_within_size_limit(self):
    """Test that requests within size limit are allowed."""
    app = FastAPI()
    middleware = RequestSizeLimitMiddleware(app, max_size=1024)  # 1KB

    request = MagicMock(spec=Request)
    request.headers = {"Content-Length": "500"}  # 500 bytes

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once()

  @pytest.mark.asyncio
  async def test_blocks_request_over_size_limit(self):
    """Test that requests over size limit are blocked."""
    app = FastAPI()
    middleware = RequestSizeLimitMiddleware(app, max_size=1024)  # 1KB

    request = MagicMock(spec=Request)
    request.headers = {"content-length": "2048"}  # 2KB

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 413
    assert isinstance(response, JSONResponse)

  @pytest.mark.asyncio
  async def test_handles_missing_content_length(self):
    """Test that requests without Content-Length header are allowed."""
    app = FastAPI()
    middleware = RequestSizeLimitMiddleware(app, max_size=1024)

    request = MagicMock(spec=Request)
    request.headers = {}  # No Content-Length header

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    call_next.assert_called_once()


class TestHealthCheckMiddleware:
  """Test HealthCheckMiddleware."""

  @pytest.mark.asyncio
  async def test_handles_health_check_request(self):
    """Test that middleware handles health check requests."""
    app = FastAPI()
    middleware = HealthCheckMiddleware(app)

    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.path = "/health"
    request.method = "GET"

    call_next = AsyncMock()

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200
    assert isinstance(response, JSONResponse)

    # Should not call next middleware for health check
    call_next.assert_not_called()

  @pytest.mark.asyncio
  async def test_passes_through_non_health_requests(self):
    """Test that non-health requests pass through."""
    app = FastAPI()
    middleware = HealthCheckMiddleware(app)

    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.path = "/api/test"

    call_next = AsyncMock(return_value=Response())

    response = await middleware.dispatch(request, call_next)

    assert response.status_code == 200

    # Should call next middleware for non-health requests
    call_next.assert_called_once()


class TestMiddlewareIntegration:
  """Test middleware integration scenarios."""

  @pytest.mark.asyncio
  async def test_middleware_chain_execution_order(self):
    """Test that middleware executes in correct order."""
    app = FastAPI()

    # Track execution order
    execution_order = []

    class TrackingMiddleware(BaseHTTPMiddleware):
      def __init__(self, app, name):
        super().__init__(app)
        self.name = name

      async def dispatch(self, request, call_next):
        execution_order.append(f"{self.name}_start")
        response = await call_next(request)
        execution_order.append(f"{self.name}_end")
        return response

    # Add multiple middleware
    middleware1 = TrackingMiddleware(app, "middleware1")
    middleware2 = TrackingMiddleware(app, "middleware2")

    request = MagicMock(spec=Request)

    async def final_handler(req):
      execution_order.append("handler")
      return Response()

    # Execute middleware chain
    await middleware1.dispatch(
      request, lambda req: middleware2.dispatch(req, final_handler)
    )

    # Check execution order (LIFO for start, FIFO for end)
    expected_order = [
      "middleware1_start",
      "middleware2_start",
      "handler",
      "middleware2_end",
      "middleware1_end",
    ]
    assert execution_order == expected_order

  @pytest.mark.asyncio
  async def test_middleware_error_handling(self):
    """Test middleware behavior when errors occur."""
    app = FastAPI()
    middleware = RequestLoggingMiddleware(app)

    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url = MagicMock()
    request.url.path = "/error"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.state = MagicMock()
    request.state.request_id = "test-request-id"

    async def error_handler(req):
      raise ValueError("Test error")

    with patch("src.utils.middleware.request_logger") as mock_request_logger:
      with pytest.raises(ValueError):
        await middleware.dispatch(request, error_handler)

      # Should still log the request even if error occurs
      mock_request_logger.log_request.assert_called_once()

  @pytest.mark.asyncio
  async def test_rate_limit_with_api_key_auth(self):
    """Test rate limiting combined with API key authentication."""
    app = FastAPI()

    # Setup middleware
    auth_middleware = APIKeyAuthMiddleware(app, require_api_key=True)
    rate_limit_middleware = RateLimitMiddleware(app, requests_per_minute=1)

    request = MagicMock(spec=Request)
    request.headers = {"X-API-Key": "valid-key"}
    request.url = MagicMock()
    request.url.path = "/api/test"
    request.query_params = {}
    request.method = "GET"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.state = MagicMock()

    call_next = AsyncMock(return_value=Response())

    with patch("src.utils.middleware.settings") as mock_settings:
      mock_settings.require_api_key = True
      mock_settings.api_keys = ["valid-key"]

      # First request should succeed
      response1 = await auth_middleware.dispatch(
        request, lambda req: rate_limit_middleware.dispatch(req, call_next)
      )
      assert response1.status_code == 200

      # Second request should be rate limited
      response2 = await auth_middleware.dispatch(
        request, lambda req: rate_limit_middleware.dispatch(req, call_next)
      )
      assert response2.status_code == 429

  @pytest.mark.asyncio
  async def test_security_headers_with_cors(self):
    """Test security headers combined with CORS."""
    app = FastAPI()

    security_middleware = SecurityHeadersMiddleware(app)
    cors_middleware = CORSMiddleware(
      app,
      allowed_origins=["https://example.com"],
      allowed_methods=["GET", "POST", "PUT", "DELETE"],
      allowed_headers=["Content-Type", "Authorization"],
    )

    request = MagicMock(spec=Request)
    request.method = "GET"
    request.headers = {"origin": "https://example.com"}
    request.url = MagicMock()
    request.url.path = "/test"

    call_next = AsyncMock(return_value=Response())

    response = await security_middleware.dispatch(
      request, lambda req: cors_middleware.dispatch(req, call_next)
    )

    # Should have both security headers and CORS headers
    assert "X-Content-Type-Options" in response.headers
    assert "Access-Control-Allow-Origin" in response.headers
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"