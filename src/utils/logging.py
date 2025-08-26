"""Logging utilities for the editor agent application."""

import logging
import sys
from datetime import datetime, UTC
from pathlib import Path

from loguru import logger as loguru_logger

from src.config.settings import get_settings


class InterceptHandler(logging.Handler):
  """Intercept standard logging and redirect to loguru."""

  def emit(self, record):
    # Get corresponding Loguru level if it exists
    try:
      level = loguru_logger.level(record.levelname).name
    except ValueError:
      level = record.levelno

    # Find caller from where originated the logged message
    frame, depth = logging.currentframe(), 2
    while frame.f_code.co_filename == logging.__file__:
      frame = frame.f_back
      depth += 1

    loguru_logger.opt(depth=depth, exception=record.exc_info).log(
      level, record.getMessage()
    )


def setup_logging() -> None:
  """Setup application logging configuration."""
  settings = get_settings()

  # Remove default loguru handler
  loguru_logger.remove()

  # Configure loguru format
  log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
  )

  # Add console handler
  loguru_logger.add(
    sys.stdout,
    format=log_format,
    level=settings.log_level,
    colorize=True,
    backtrace=True,
    diagnose=True,
  )

  # Add file handler if log file is specified
  if settings.log_file:
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    loguru_logger.add(
      str(log_path),
      format=log_format,
      level=settings.log_level,
      rotation=settings.log_rotation,
      retention=settings.log_retention,
      compression="zip",
      backtrace=True,
      diagnose=True,
    )

  # Intercept standard logging
  logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

  # Set levels for specific loggers
  for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
    logging.getLogger(logger_name).handlers = [InterceptHandler()]
    logging.getLogger(logger_name).setLevel(logging.INFO)

  # Reduce noise from some libraries
  logging.getLogger("httpx").setLevel(logging.WARNING)
  logging.getLogger("httpcore").setLevel(logging.WARNING)
  logging.getLogger("urllib3").setLevel(logging.WARNING)

  loguru_logger.info("Logging setup completed")


def get_logger(name: str):
  """Get a logger instance for the given name."""
  return loguru_logger.bind(name=name)


class LoggerMixin:
  """Mixin class to add logging capabilities to any class."""

  @property
  def logger(self):
    """Get logger instance for this class."""
    return get_logger(self.__class__.__name__)


def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
  """Decorator to log function calls."""

  def decorator(func):
    def wrapper(*args, **kwargs):
      logger = get_logger(func.__module__)
      logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")

      start_time = datetime.now(UTC)
      try:
        result = func(*args, **kwargs)
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        logger.debug(f"{func_name} completed in {duration:.3f}s")
        return result
      except Exception as e:
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        logger.error(f"{func_name} failed after {duration:.3f}s: {str(e)}")
        raise

    return wrapper

  return decorator


def log_async_function_call(func_name: str):
  """Decorator to log async function calls."""

  def decorator(func):
    async def wrapper(*args, **kwargs):
      logger = get_logger(func.__module__)
      logger.debug(f"Calling async {func_name} with args={args}, kwargs={kwargs}")

      start_time = datetime.now(UTC)
      try:
        result = await func(*args, **kwargs)
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        logger.debug(f"Async {func_name} completed in {duration:.3f}s")
        return result
      except Exception as e:
        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()
        logger.error(f"Async {func_name} failed after {duration:.3f}s: {str(e)}")
        raise

    return wrapper

  return decorator


class RequestLogger:
  """Logger for HTTP requests."""

  def __init__(self):
    self.logger = get_logger("request")

  def log_request(self, method: str, url: str, headers: dict = None, body: str = None):
    """Log incoming HTTP request."""
    self.logger.info(f"Request: {method} {url}")
    if headers:
      self.logger.debug(f"Headers: {headers}")
    if body:
      self.logger.debug(f"Body: {body}")

  def log_response(self, status_code: int, response_time: float, size: int = None):
    """Log HTTP response."""
    self.logger.info(f"Response: {status_code} ({response_time:.3f}s)")
    if size:
      self.logger.debug(f"Response size: {size} bytes")


class AgentLogger:
  """Logger for agent operations."""

  def __init__(self):
    self.logger = get_logger("agent")

  def log_task_start(self, task_id: str, task_description: str):
    """Log the start of an agent task."""
    self.logger.info(f"Task started: {task_id} - {task_description}")

  def log_task_step(self, task_id: str, step: str, details: str = None):
    """Log a step in an agent task."""
    message = f"Task {task_id} - Step: {step}"
    if details:
      message += f" - {details}"
    self.logger.info(message)

  def log_task_complete(self, task_id: str, duration: float, success: bool = True):
    """Log the completion of an agent task."""
    status = "completed" if success else "failed"
    self.logger.info(f"Task {status}: {task_id} ({duration:.3f}s)")

  def log_tool_usage(self, tool_name: str, parameters: dict, result: dict):
    """Log tool usage by the agent."""
    success = result.get("success", False)
    status = "success" if success else "failed"
    self.logger.info(f"Tool {tool_name} - {status}")
    self.logger.debug(f"Tool parameters: {parameters}")
    if not success and "error" in result:
      self.logger.error(f"Tool error: {result['error']}")


class PerformanceLogger:
  """Logger for performance metrics."""

  def __init__(self):
    self.logger = get_logger("performance")

  def log_execution_time(self, operation: str, duration: float, context: dict = None):
    """Log execution time for an operation."""
    message = f"Performance: {operation} took {duration:.3f}s"
    if context:
      message += f" (context: {context})"
    self.logger.info(message)

  def log_memory_usage(self, operation: str, memory_mb: float):
    """Log memory usage for an operation."""
    self.logger.info(f"Memory: {operation} used {memory_mb:.2f}MB")

  def log_api_call(
    self, provider: str, model: str, tokens_used: int, cost: float = None
  ):
    """Log API call metrics."""
    message = f"API Call: {provider}/{model} - {tokens_used} tokens"
    if cost:
      message += f" (${cost:.4f})"
    self.logger.info(message)


# Global logger instances
request_logger = RequestLogger()
agent_logger = AgentLogger()
performance_logger = PerformanceLogger()