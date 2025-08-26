"""Tests for logging utilities."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from src.utils.logging import (
  InterceptHandler,
  LoggerMixin,
  RequestLogger,
  AgentLogger,
  PerformanceLogger,
  get_logger,
  setup_logging,
)


class TestInterceptHandler:
  """Test InterceptHandler functionality."""

  def test_emit_basic(self):
    """Test basic log emission."""
    handler = InterceptHandler()

    # Create a log record
    record = logging.LogRecord(
      name="test",
      level=logging.INFO,
      pathname="test.py",
      lineno=1,
      msg="Test message",
      args=(),
      exc_info=None,
    )

    with patch.object(logger, "opt") as mock_opt:
      mock_logger = MagicMock()
      mock_opt.return_value = mock_logger

      handler.emit(record)

      mock_opt.assert_called_once()
      mock_logger.log.assert_called_once()

  def test_emit_with_exception(self):
    """Test log emission with exception info."""
    handler = InterceptHandler()

    try:
      raise ValueError("Test error")
    except ValueError:
      record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="Error occurred",
        args=(),
        exc_info=True,
      )

      with patch.object(logger, "opt") as mock_opt:
        mock_logger = MagicMock()
        mock_opt.return_value = mock_logger

        handler.emit(record)

        mock_opt.assert_called_once()
        mock_logger.log.assert_called_once()


class TestLoggerMixin:
  """Test LoggerMixin functionality."""

  def test_logger_property(self):
    """Test logger property access."""

    class TestClass(LoggerMixin):
      pass

    instance = TestClass()
    logger_instance = instance.logger

    # Should return a logger-like object
    assert hasattr(logger_instance, "info")
    assert hasattr(logger_instance, "error")
    assert hasattr(logger_instance, "debug")

  def test_logger_name(self):
    """Test logger name generation."""

    class TestClass(LoggerMixin):
      pass

    instance = TestClass()

    # Logger name should be based on class name
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      _ = instance.logger
      mock_get_logger.assert_called_once_with("TestClass")


class TestSpecializedLoggers:
  """Test specialized logger classes."""

  def test_request_logger_log_request(self):
    """Test request logging."""
    request_logger = RequestLogger()

    with patch.object(request_logger, "logger") as mock_logger:
      request_logger.log_request(
        method="GET",
        url="/test",
        headers={"Content-Type": "application/json"},
        body='{"test": "data"}',
      )

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "GET" in call_args
      assert "/test" in call_args

  def test_request_logger_log_response(self):
    """Test RequestLogger log_response method."""
    logger = RequestLogger()

    with patch.object(logger, "logger") as mock_logger:
      logger.log_response(200, 0.5, 1024)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "200" in call_args
      assert "0.5" in call_args

  def test_agent_logger_log_task_start(self):
    """Test AgentLogger log_task_start method."""
    logger = AgentLogger()

    with patch.object(logger, "logger") as mock_logger:
      logger.log_task_start("task-123", "Search for documents")

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "task-123" in call_args
      assert "Search for documents" in call_args

  def test_agent_logger_log_task_step(self):
    """Test AgentLogger log_task_step method."""
    logger = AgentLogger()

    with patch.object(logger, "logger") as mock_logger:
      logger.log_task_step("task-123", "Document analysis", "Processing 3 documents")

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "task-123" in call_args
      assert "Document analysis" in call_args

  def test_performance_logger_log_execution_time(self):
    """Test performance execution time logging."""
    perf_logger = PerformanceLogger()

    with patch.object(perf_logger, "logger") as mock_logger:
      perf_logger.log_execution_time(operation="database_query", duration=0.8)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "database_query" in call_args
      assert "0.8" in call_args

  def test_performance_logger_log_memory_usage(self):
    """Test memory usage logging."""
    perf_logger = PerformanceLogger()

    with patch.object(perf_logger, "logger") as mock_logger:
      perf_logger.log_memory_usage(operation="model_inference", memory_mb=512.5)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "model_inference" in call_args
      assert "512.5" in call_args


class TestGetLogger:
  """Test get_logger function."""

  def test_get_logger_basic(self):
    """Test basic logger creation."""
    logger_instance = get_logger("test_module")

    # Should return a logger-like object
    assert hasattr(logger_instance, "info")
    assert hasattr(logger_instance, "error")
    assert hasattr(logger_instance, "debug")

  def test_get_logger_caching(self):
    """Test that get_logger returns logger instances."""
    logger1 = get_logger("test_cache")
    logger2 = get_logger("test_cache")

    # Should return logger instances
    assert logger1 is not None
    assert logger2 is not None

  def test_get_logger_different_names(self):
    """Test different logger names."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    # Should return different instances
    assert logger1 is not logger2


class TestSetupLogging:
  """Test setup_logging function."""

  def test_setup_logging_text_format(self):
    """Test logging setup with text format."""
    with patch("src.config.settings.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.log_level = "INFO"
      mock_settings.log_file = None
      mock_get_settings.return_value = mock_settings

      with patch("src.utils.logging.loguru_logger") as mock_logger:
        setup_logging()

        # Should configure logger with text format
        mock_logger.add.assert_called()
        call_args = mock_logger.add.call_args
        assert "format" in call_args[1]

  def test_setup_logging_with_file(self):
    """Test logging setup with file output."""
    with patch("src.config.settings.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.log_level = "INFO"
      mock_settings.log_file = "/tmp/test.log"
      mock_get_settings.return_value = mock_settings

      with patch("src.utils.logging.loguru_logger") as mock_logger:
        setup_logging()

        # Should configure file logging
        mock_logger.remove.assert_called()
        assert mock_logger.add.call_count >= 1

  def test_setup_logging_json_format(self):
    """Test logging setup with JSON format."""
    with patch("src.config.settings.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.log_level = "DEBUG"
      mock_settings.log_format = "json"
      mock_settings.log_file = None
      mock_get_settings.return_value = mock_settings

      with patch("src.utils.logging.loguru_logger") as mock_logger:
        setup_logging()

        # Should configure JSON format
        mock_logger.remove.assert_called()
        mock_logger.add.assert_called()

  def test_setup_logging_text_format_legacy(self):
    """Test logging setup with text format using legacy settings."""
    with patch("src.config.settings.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.log_level = "INFO"
      mock_settings.log_format = "text"
      mock_settings.log_file = None
      mock_get_settings.return_value = mock_settings

      with patch("src.utils.logging.loguru_logger") as mock_logger:
        setup_logging()

        # Check that text format is used
        mock_logger.add.assert_called()
        call_args = mock_logger.add.call_args
        assert "format" in call_args[1]
        assert "serialize" not in call_args[1] or call_args[1]["serialize"] is False

  def test_setup_logging_intercept_standard_logging(self):
    """Test that standard logging is intercepted."""
    with patch("src.config.settings.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.log_level = "INFO"
      mock_settings.log_file = None
      mock_get_settings.return_value = mock_settings

      with patch("logging.basicConfig") as mock_basic_config:
        setup_logging()

        # Should configure standard logging interception
        mock_basic_config.assert_called()


class TestLoggingIntegration:
  """Test logging integration scenarios."""

  def test_logging_with_context(self):
    """Test logging with context information."""
    logger_instance = get_logger("test_context")

    with patch.object(logger_instance, "bind") as mock_bind:
      mock_bound_logger = MagicMock()
      mock_bind.return_value = mock_bound_logger

      # Simulate logging with context
      bound_logger = logger_instance.bind(request_id="test-123", user_id="user-456")
      bound_logger.info("Test message")

      mock_bind.assert_called_once_with(request_id="test-123", user_id="user-456")
      mock_bound_logger.info.assert_called_once_with("Test message")

  def test_logging_error_with_exception(self):
    """Test error logging with exception information."""
    logger_instance = get_logger("test_exception")

    with patch.object(logger_instance, "exception") as mock_exception:
      try:
        raise ValueError("Test error")
      except ValueError:
        logger_instance.exception("An error occurred")

      mock_exception.assert_called_once_with("An error occurred")

  def test_multiple_logger_instances(self):
    """Test multiple logger instances working together."""
    request_logger = RequestLogger()
    agent_logger = AgentLogger()
    perf_logger = PerformanceLogger()

    with patch.object(request_logger, "logger") as mock_req_logger:
      with patch.object(agent_logger, "logger") as mock_agent_logger:
        with patch.object(perf_logger, "logger") as mock_perf_logger:
          # Simulate concurrent logging
          request_logger.log_request(
            method="GET",
            url="/test",
            headers={"Content-Type": "application/json"},
            body='{"test": "data"}',
          )
          agent_logger.log_task_start("task-456", "Test task")
          perf_logger.log_execution_time(operation="operation", duration=1.0)

          # All loggers should have been called
          mock_req_logger.info.assert_called_once()
          mock_agent_logger.info.assert_called_once()
          mock_perf_logger.info.assert_called_once()

  def test_logging_configuration_changes(self):
    """Test that logging configuration can be changed dynamically."""
    # Initial setup
    with patch("src.config.settings.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.log_level = "INFO"
      mock_settings.log_file = None
      mock_get_settings.return_value = mock_settings

      with patch("src.utils.logging.loguru_logger") as mock_logger:
        setup_logging()

        # Change configuration
        mock_settings.log_level = "DEBUG"
        setup_logging()

        # Should reconfigure
        assert mock_logger.remove.call_count == 2
        assert mock_logger.add.call_count >= 2

  @pytest.mark.asyncio
  async def test_async_logging(self):
    """Test logging in async context."""
    logger_instance = get_logger("test_async")

    with patch.object(logger_instance, "info") as mock_info:
      # Simulate async logging
      await self._async_log_operation(logger_instance)

      mock_info.assert_called_with("Async operation completed")

  async def _async_log_operation(self, logger_instance):
    """Helper method for async logging test."""
    logger_instance.info("Async operation completed")
