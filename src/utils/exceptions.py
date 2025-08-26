"""Custom exceptions and error handling for the editor agent application."""

from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EditorAgentException(Exception):
  """Base exception class for editor agent."""

  def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None):
    self.message = message
    self.code = code or self.__class__.__name__
    self.details = details or {}
    super().__init__(self.message)

  def to_dict(self) -> Dict[str, Any]:
    """Convert exception to dictionary."""
    return {
      "error": {"code": self.code, "message": self.message, "details": self.details}
    }


class ValidationError(EditorAgentException):
  """Raised when input validation fails."""

  pass


class AuthenticationError(EditorAgentException):
  """Raised when authentication fails."""

  pass


class AuthorizationError(EditorAgentException):
  """Raised when authorization fails."""

  pass


class ResourceNotFoundError(EditorAgentException):
  """Raised when a requested resource is not found."""

  pass


class ResourceConflictError(EditorAgentException):
  """Raised when a resource conflict occurs."""

  pass


class ExternalServiceError(EditorAgentException):
  """Raised when an external service call fails."""

  pass


class ModelError(EditorAgentException):
  """Raised when AI model operations fail."""

  pass


class WorkflowError(EditorAgentException):
  """Raised when workflow execution fails."""

  pass


class ToolExecutionError(EditorAgentException):
  """Raised when tool execution fails."""

  pass


class FileSystemError(EditorAgentException):
  """Raised when file system operations fail."""

  pass


class ConfigurationError(EditorAgentException):
  """Raised when configuration is invalid."""

  pass


class RateLimitError(EditorAgentException):
  """Raised when rate limit is exceeded."""

  pass


class TimeoutError(EditorAgentException):
  """Raised when an operation times out."""

  pass


# HTTP Exception mappings
EXCEPTION_STATUS_MAPPING = {
  ValidationError: status.HTTP_400_BAD_REQUEST,
  AuthenticationError: status.HTTP_401_UNAUTHORIZED,
  AuthorizationError: status.HTTP_403_FORBIDDEN,
  ResourceNotFoundError: status.HTTP_404_NOT_FOUND,
  ResourceConflictError: status.HTTP_409_CONFLICT,
  RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
  TimeoutError: status.HTTP_408_REQUEST_TIMEOUT,
  ExternalServiceError: status.HTTP_502_BAD_GATEWAY,
  ModelError: status.HTTP_503_SERVICE_UNAVAILABLE,
  WorkflowError: status.HTTP_500_INTERNAL_SERVER_ERROR,
  ToolExecutionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
  FileSystemError: status.HTTP_500_INTERNAL_SERVER_ERROR,
  ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def create_http_exception(exc: EditorAgentException) -> HTTPException:
  """Create HTTPException from EditorAgentException."""
  status_code = EXCEPTION_STATUS_MAPPING.get(
    type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
  )
  return HTTPException(status_code=status_code, detail=exc.to_dict())


async def editor_agent_exception_handler(
  request: Request, exc: EditorAgentException
) -> JSONResponse:
  """Global exception handler for EditorAgentException."""
  status_code = EXCEPTION_STATUS_MAPPING.get(
    type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
  )

  logger.error(
    f"EditorAgentException: {exc.code} - {exc.message}",
    extra={
      "exception_type": type(exc).__name__,
      "status_code": status_code,
      "details": exc.details,
      "request_url": str(request.url),
      "request_method": request.method,
    },
  )

  return JSONResponse(status_code=status_code, content=exc.to_dict())


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
  """Global exception handler for HTTPException."""
  logger.warning(
    f"HTTPException: {exc.status_code} - {exc.detail}",
    extra={
      "status_code": exc.status_code,
      "request_url": str(request.url),
      "request_method": request.method,
    },
  )

  return JSONResponse(
    status_code=exc.status_code,
    content={
      "error": {"code": f"HTTP_{exc.status_code}", "message": exc.detail, "details": {}}
    },
  )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
  """Global exception handler for unexpected exceptions."""
  logger.error(
    f"Unexpected exception: {type(exc).__name__} - {str(exc)}",
    extra={
      "exception_type": type(exc).__name__,
      "request_url": str(request.url),
      "request_method": request.method,
    },
    exc_info=True,
  )

  return JSONResponse(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    content={
      "error": {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred",
        "details": {"exception_type": type(exc).__name__},
      }
    },
  )


class ErrorContext:
  """Context manager for handling errors with additional context."""

  def __init__(self, operation: str, **context):
    self.operation = operation
    self.context = context
    self.logger = get_logger("error_context")

  def __enter__(self):
    self.logger.debug(f"Starting operation: {self.operation}", extra=self.context)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
      self.logger.error(
        f"Operation failed: {self.operation} - {exc_val}",
        extra={
          **self.context,
          "exception_type": exc_type.__name__,
          "exception_message": str(exc_val),
        },
        exc_info=True,
      )
    else:
      self.logger.debug(f"Operation completed: {self.operation}", extra=self.context)
    return False  # Don't suppress exceptions


def handle_external_service_error(service_name: str, operation: str):
  """Decorator to handle external service errors."""

  def decorator(func):
    def wrapper(*args, **kwargs):
      try:
        return func(*args, **kwargs)
      except Exception as e:
        logger.error(
          f"External service error: {service_name}.{operation} - {str(e)}",
          extra={
            "service": service_name,
            "operation": operation,
            "args": str(args),
            "kwargs": str(kwargs),
          },
          exc_info=True,
        )
        raise ExternalServiceError(
          f"Failed to {operation} with {service_name}: {str(e)}",
          code=f"{service_name.upper()}_{operation.upper()}_ERROR",
          details={
            "service": service_name,
            "operation": operation,
            "original_error": str(e),
          },
        )

    return wrapper

  return decorator


def handle_async_external_service_error(service_name: str, operation: str):
  """Decorator to handle external service errors for async functions."""

  def decorator(func):
    async def wrapper(*args, **kwargs):
      try:
        return await func(*args, **kwargs)
      except Exception as e:
        logger.error(
          f"External service error: {service_name}.{operation} - {str(e)}",
          extra={
            "service": service_name,
            "operation": operation,
            "args": str(args),
            "kwargs": str(kwargs),
          },
          exc_info=True,
        )
        raise ExternalServiceError(
          f"Failed to {operation} with {service_name}: {str(e)}",
          code=f"{service_name.upper()}_{operation.upper()}_ERROR",
          details={
            "service": service_name,
            "operation": operation,
            "original_error": str(e),
          },
        )

    return wrapper

  return decorator


def validate_input(validation_func, error_message: str = None):
  """Decorator to validate function input."""

  def decorator(func):
    def wrapper(*args, **kwargs):
      try:
        validation_func(*args, **kwargs)
        return func(*args, **kwargs)
      except Exception as e:
        message = error_message or f"Input validation failed: {str(e)}"
        raise ValidationError(
          message,
          code="INPUT_VALIDATION_ERROR",
          details={
            "validation_error": str(e),
            "args": str(args),
            "kwargs": str(kwargs),
          },
        )

    return wrapper

  return decorator


def retry_on_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
  """Decorator to retry function on specific errors."""
  import asyncio
  import time

  def decorator(func):
    if asyncio.iscoroutinefunction(func):

      async def async_wrapper(*args, **kwargs):
        last_exception = None
        current_delay = delay

        for attempt in range(max_retries + 1):
          try:
            return await func(*args, **kwargs)
          except (ExternalServiceError, TimeoutError, ModelError) as e:
            last_exception = e
            if attempt < max_retries:
              logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {current_delay}s: {str(e)}"
              )
              await asyncio.sleep(current_delay)
              current_delay *= backoff
            else:
              logger.error(f"All {max_retries + 1} attempts failed")
              raise
          except Exception:
            # Don't retry on other types of exceptions
            raise

        if last_exception:
          raise last_exception

      return async_wrapper
    else:

      def sync_wrapper(*args, **kwargs):
        last_exception = None
        current_delay = delay

        for attempt in range(max_retries + 1):
          try:
            return func(*args, **kwargs)
          except (ExternalServiceError, TimeoutError, ModelError) as e:
            last_exception = e
            if attempt < max_retries:
              logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {current_delay}s: {str(e)}"
              )
              time.sleep(current_delay)
              current_delay *= backoff
            else:
              logger.error(f"All {max_retries + 1} attempts failed")
              raise
          except Exception:
            # Don't retry on other types of exceptions
            raise

        if last_exception:
          raise last_exception

      return sync_wrapper

  return decorator
