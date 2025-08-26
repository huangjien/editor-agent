"""Tests for exception utilities."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from src.utils.exceptions import (
  EditorAgentException,
  ValidationError,
  ResourceNotFoundError,
  AuthorizationError,
  ExternalServiceError,
  ConfigurationError,
  RateLimitError,
  AuthenticationError,
  EXCEPTION_STATUS_MAPPING,
  editor_agent_exception_handler,
  http_exception_handler,
  general_exception_handler,
  ErrorContext,
  handle_external_service_error,
  handle_async_external_service_error,
  validate_input,
  retry_on_error,
)


class TestEditorAgentException:
  """Test base EditorAgentException class."""

  def test_basic_exception(self):
    """Test basic exception creation."""
    exc = EditorAgentException("Test error")

    assert str(exc) == "Test error"
    assert exc.message == "Test error"
    assert exc.code == "EditorAgentException"
    assert exc.details == {}

  def test_exception_with_details(self):
    """Test exception with details."""
    details = {"field": "value", "context": "test"}
    exc = EditorAgentException("Test error", details=details)

    assert exc.details == details

  def test_exception_with_custom_code(self):
    """Test exception with custom error code."""
    exc = EditorAgentException("Test error", code="CUSTOM_ERROR")

    assert exc.code == "CUSTOM_ERROR"


class TestSpecificExceptions:
  """Test specific exception classes."""

  def test_validation_error(self):
    """Test ValidationError."""
    exc = ValidationError("Invalid input")

    assert str(exc) == "Invalid input"
    assert exc.code == "ValidationError"

  def test_resource_not_found_error(self):
    """Test ResourceNotFoundError."""
    exc = ResourceNotFoundError("File not found")

    assert str(exc) == "File not found"
    assert exc.code == "ResourceNotFoundError"

  def test_authorization_error(self):
    """Test AuthorizationError."""
    exc = AuthorizationError("Access denied")

    assert str(exc) == "Access denied"
    assert exc.code == "AuthorizationError"

  def test_external_service_error(self):
    """Test ExternalServiceError."""
    exc = ExternalServiceError("API unavailable")

    assert str(exc) == "API unavailable"
    assert exc.code == "ExternalServiceError"

  def test_configuration_error(self):
    """Test ConfigurationError."""
    exc = ConfigurationError("Invalid config")

    assert str(exc) == "Invalid config"
    assert exc.code == "ConfigurationError"

  def test_rate_limit_error(self):
    """Test RateLimitError."""
    exc = RateLimitError("Rate limit exceeded")

    assert str(exc) == "Rate limit exceeded"
    assert exc.code == "RateLimitError"

  def test_authentication_error(self):
    """Test AuthenticationError."""
    exc = AuthenticationError("Invalid credentials")

    assert str(exc) == "Invalid credentials"
    assert exc.code == "AuthenticationError"


class TestExceptionStatusMap:
  """Test exception to status code mapping."""

  def test_status_map_completeness(self):
    """Test that all exception types have status codes."""
    exception_types = [
      ValidationError,
      ResourceNotFoundError,
      AuthorizationError,
      ExternalServiceError,
      ConfigurationError,
      RateLimitError,
      AuthenticationError,
    ]

    for exc_type in exception_types:
      assert exc_type in EXCEPTION_STATUS_MAPPING

  def test_status_codes(self):
    """Test that status codes are correctly mapped."""
    # Test specific mappings
    assert EXCEPTION_STATUS_MAPPING[ValidationError] == 400
    assert EXCEPTION_STATUS_MAPPING[ResourceNotFoundError] == 404
    assert EXCEPTION_STATUS_MAPPING[AuthorizationError] == 403
    assert EXCEPTION_STATUS_MAPPING[ExternalServiceError] == 502
    assert EXCEPTION_STATUS_MAPPING[ConfigurationError] == 500
    assert EXCEPTION_STATUS_MAPPING[RateLimitError] == 429
    assert EXCEPTION_STATUS_MAPPING[AuthenticationError] == 401


class TestExceptionHandlers:
  """Test exception handler functions."""

  @pytest.mark.asyncio
  async def test_editor_agent_exception_handler(self):
    """Test EditorAgentException handler."""
    request = MagicMock(spec=Request)
    exc = ValidationError("Invalid input", details={"field": "test"})

    response = await editor_agent_exception_handler(request, exc)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400

    # Check response content
    content = response.body.decode()
    assert "Invalid input" in content
    assert "ValidationError" in content

  @pytest.mark.asyncio
  async def test_http_exception_handler(self):
    """Test HTTPException handler."""
    request = MagicMock(spec=Request)
    exc = HTTPException(status_code=404, detail="Not found")

    response = await http_exception_handler(request, exc)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 404

    content = response.body.decode()
    assert "Not found" in content

  @pytest.mark.asyncio
  async def test_general_exception_handler(self):
    """Test general exception handler."""
    request = MagicMock(spec=Request)
    exc = ValueError("Unexpected error")

    with patch("src.utils.exceptions.logger") as mock_logger:
      response = await general_exception_handler(request, exc)

      assert isinstance(response, JSONResponse)
      assert response.status_code == 500

      # Should log the error
      mock_logger.error.assert_called_once()

      content = response.body.decode()
      assert "An unexpected error occurred" in content


class TestErrorContextDecorator:
  """Test ErrorContext context manager."""

  def test_error_context_success(self):
    """Test ErrorContext with successful function."""

    def test_function():
      with ErrorContext("test_operation"):
        return "success"

    result = test_function()
    assert result == "success"

  def test_error_context_with_exception(self):
    """Test ErrorContext with exception."""

    def test_function():
      with ErrorContext("test_operation"):
        raise ValueError("Test error")

    with pytest.raises(ValueError) as exc_info:
      test_function()

    assert str(exc_info.value) == "Test error"

  def test_error_context_preserves_editor_agent_exception(self):
    """Test that EditorAgentException is preserved."""

    def test_function():
      with ErrorContext("test_operation"):
        raise ValidationError("Validation failed")

    with pytest.raises(ValidationError) as exc_info:
      test_function()

    assert str(exc_info.value) == "Validation failed"

  def test_error_context_with_custom_exception_type(self):
    """Test ErrorContext with custom exception type."""

    def test_function():
      with ErrorContext("test_operation", exception_type=ConfigurationError):
        raise ValueError("Test error")

    with pytest.raises(ValueError) as exc_info:
      test_function()

    assert str(exc_info.value) == "Test error"


class TestExternalServiceErrorHandlers:
  """Test external service error handling decorators."""

  def test_handle_external_service_error_success(self):
    """Test successful external service call."""

    @handle_external_service_error("test_service", "test_operation")
    def test_function():
      return "success"

    result = test_function()
    assert result == "success"

  def test_handle_external_service_error_with_exception(self):
    """Test external service error handling."""

    @handle_external_service_error("test_service", "test_operation")
    def test_function():
      raise ConnectionError("Connection failed")

    with pytest.raises(ExternalServiceError) as exc_info:
      test_function()

    assert "test_service" in str(exc_info.value)
    assert exc_info.value.details["service"] == "test_service"

  @pytest.mark.asyncio
  async def test_handle_async_external_service_error_success(self):
    """Test successful async external service call."""

    @handle_async_external_service_error("test_service", "test_operation")
    async def test_function():
      return "success"

    result = await test_function()
    assert result == "success"

  @pytest.mark.asyncio
  async def test_handle_async_external_service_error_with_exception(self):
    """Test async external service error handling."""

    @handle_async_external_service_error("test_service", "test_operation")
    async def test_function():
      raise TimeoutError("Request timeout")

    with pytest.raises(ExternalServiceError) as exc_info:
      await test_function()

    assert "test_service" in str(exc_info.value)
    assert exc_info.value.details["service"] == "test_service"


class TestValidateInputDecorator:
  """Test validate_input decorator."""

  def test_validate_input_success(self):
    """Test successful input validation."""

    def validation_func(*args, **kwargs):
      value = args[0] if args else kwargs.get("input_param")
      if not isinstance(value, str) or len(value) == 0:
        raise ValueError("Input must be non-empty string")

    @validate_input(validation_func, "Input must be non-empty string")
    def test_function(input_param):
      return f"Processed: {input_param}"

    result = test_function("valid_input")
    assert result == "Processed: valid_input"

  def test_validate_input_failure(self):
    """Test input validation failure."""

    def validation_func(*args, **kwargs):
      value = args[0] if args else kwargs.get("input_param")
      if not isinstance(value, str) or len(value) == 0:
        raise ValueError("Input must be non-empty string")

    @validate_input(validation_func, "Input must be non-empty string")
    def test_function(input_param):
      return f"Processed: {input_param}"

    with pytest.raises(ValidationError) as exc_info:
      test_function("")

    assert "Input must be non-empty string" in str(exc_info.value)

  def test_validate_input_missing_parameter(self):
    """Test validation with missing parameter."""

    def validation_func(*args, **kwargs):
      value = args[0] if args else kwargs.get("input_param")
      if not isinstance(value, str) or len(value) == 0:
        raise ValueError("Input must be non-empty string")

    @validate_input(validation_func, "Input must be non-empty string")
    def test_function(input_param):
      return f"Processed: {input_param}"

    with pytest.raises(ValidationError) as exc_info:
      test_function(123)  # Pass invalid type

    assert "Input must be non-empty string" in str(exc_info.value)


class TestRetryDecorator:
  """Test retry_on_error decorator."""

  def test_retry_success_first_attempt(self):
    """Test successful function on first attempt."""
    call_count = 0

    @retry_on_error(max_retries=3, delay=0.1)
    def test_function():
      nonlocal call_count
      call_count += 1
      return "success"

    result = test_function()
    assert result == "success"
    assert call_count == 1

  def test_retry_success_after_failures(self):
    """Test successful function after some failures."""
    call_count = 0

    @retry_on_error(max_retries=3, delay=0.1)
    def test_function():
      nonlocal call_count
      call_count += 1
      if call_count < 3:
        raise ExternalServiceError("Temporary failure")
      return "success"

    result = test_function()
    assert result == "success"
    assert call_count == 3

  def test_retry_max_attempts_exceeded(self):
    """Test retry with max attempts exceeded."""
    call_count = 0

    @retry_on_error(max_retries=2, delay=0.1)
    def test_function():
      nonlocal call_count
      call_count += 1
      raise ExternalServiceError("Persistent failure")

    with pytest.raises(ExternalServiceError):
      test_function()

    assert call_count == 3

  def test_retry_with_specific_exceptions(self):
    """Test retry with specific exception types."""
    call_count = 0

    @retry_on_error(max_retries=3, delay=0.1)
    def test_function():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ExternalServiceError("Retryable error")
      elif call_count == 2:
        raise ValueError("Non-retryable error")
      return "success"

    with pytest.raises(ValueError):
      test_function()

    assert call_count == 2

  @pytest.mark.asyncio
  async def test_retry_async_function(self):
    """Test retry with async function."""
    call_count = 0

    @retry_on_error(max_retries=3, delay=0.1)
    async def test_function():
      nonlocal call_count
      call_count += 1
      if call_count < 2:
        raise ExternalServiceError("Temporary failure")
      return "success"

    result = await test_function()
    assert result == "success"
    assert call_count == 2


class TestExceptionIntegration:
  """Test exception handling integration scenarios."""

  def test_nested_decorators(self):
    """Test multiple decorators working together."""
    call_count = 0

    @handle_external_service_error("test_service", "test_operation")
    @retry_on_error(max_retries=2, delay=0.1)
    def test_function():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ExternalServiceError("Temporary failure")
      return "success"

    result = test_function()
    assert result == "success"
    assert call_count == 2

  def test_validation_with_error_context(self):
    """Test validation decorator with error context."""

    def validation_func(*args, **kwargs):
      value = args[0] if args else kwargs.get("number")
      if not isinstance(value, int) or value <= 0:
        raise ValueError("Number must be positive integer")

    @validate_input(validation_func, "Number must be positive integer")
    def test_function(number):
      return number * 2

    # Test success
    result = test_function(5)
    assert result == 10

    # Test validation failure
    with pytest.raises(ValidationError):
      test_function(-1)

  def test_external_service_with_retry(self):
    """Test external service decorator with retry."""
    call_count = 0

    @handle_external_service_error("api_service", "test_operation")
    @retry_on_error(max_retries=2, delay=0.1)
    def test_function():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ExternalServiceError("Network error")
      return "api_response"

    result = test_function()
    assert result == "api_response"
    assert call_count == 2

  @pytest.mark.asyncio
  async def test_exception_handler_with_request_context(self):
    """Test exception handlers with request context."""
    request = MagicMock(spec=Request)
    request.url = "/test/endpoint"
    request.method = "POST"

    exc = ValidationError(
      "Invalid request data", details={"field": "email", "value": "invalid"}
    )

    with patch("src.utils.exceptions.logger") as mock_logger:
      response = await editor_agent_exception_handler(request, exc)

      assert response.status_code == 400

      # Should log with request context
      mock_logger.error.assert_called_once()
      call_args = mock_logger.error.call_args
      # Check the extra fields contain request context
      extra = call_args[1]["extra"]
      assert extra["request_method"] == "POST"
      assert extra["request_url"] == "/test/endpoint"

  def test_exception_chaining(self):
    """Test exception chaining and context preservation."""

    def outer_function():
      with ErrorContext("outer_operation"):

        def inner_function():
          with ErrorContext("inner_operation"):
            raise ValueError("Original error")

        inner_function()

    with pytest.raises(ValueError) as exc_info:
      outer_function()

    # Should preserve the original exception
    assert "Original error" in str(exc_info.value)
