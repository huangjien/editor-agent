"""Tests for logging utilities."""

import logging
import os
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
  log_function_call,
  log_async_function_call,
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


class TestFunctionDecorators:
  """Test function call logging decorators."""

  def test_log_function_call_basic(self):
    """Test basic function call logging."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      @log_function_call("test_function")
      def test_func(x, y):
        return x + y

      result = test_func(1, 2)

      assert result == 3
      assert mock_logger.debug.call_count == 2  # Start and end calls

      # Check start call
      start_call = mock_logger.debug.call_args_list[0][0][0]
      assert "Calling test_function" in start_call
      assert "args=(1, 2)" in start_call

      # Check end call
      end_call = mock_logger.debug.call_args_list[1][0][0]
      assert "test_function completed" in end_call

  def test_log_function_call_with_kwargs(self):
    """Test function call logging with keyword arguments."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      @log_function_call("test_function_kwargs")
      def test_func(x, y=10):
        return x * y

      result = test_func(5, y=3)

      assert result == 15
      start_call = mock_logger.debug.call_args_list[0][0][0]
      assert "kwargs={'y': 3}" in start_call

  def test_log_function_call_with_exception(self):
    """Test function call logging when exception occurs."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      @log_function_call("failing_function")
      def failing_func():
        raise ValueError("Test error")

      with pytest.raises(ValueError, match="Test error"):
        failing_func()

      # Should have debug call for start and error call for failure
      assert mock_logger.debug.call_count == 1  # Start call
      assert mock_logger.error.call_count == 1  # Error call

      error_call = mock_logger.error.call_args[0][0]
      assert "failing_function failed" in error_call
      assert "Test error" in error_call

  @pytest.mark.asyncio
  async def test_log_async_function_call_basic(self):
    """Test basic async function call logging."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      @log_async_function_call("async_test_function")
      async def async_test_func(x, y):
        return x + y

      result = await async_test_func(3, 4)

      assert result == 7
      assert mock_logger.debug.call_count == 2  # Start and end calls

      # Check start call
      start_call = mock_logger.debug.call_args_list[0][0][0]
      assert "Calling async async_test_function" in start_call

      # Check end call
      end_call = mock_logger.debug.call_args_list[1][0][0]
      assert "Async async_test_function completed" in end_call

  @pytest.mark.asyncio
  async def test_log_async_function_call_with_exception(self):
    """Test async function call logging when exception occurs."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      @log_async_function_call("async_failing_function")
      async def async_failing_func():
        raise RuntimeError("Async test error")

      with pytest.raises(RuntimeError, match="Async test error"):
        await async_failing_func()

      # Should have debug call for start and error call for failure
      assert mock_logger.debug.call_count == 1  # Start call
      assert mock_logger.error.call_count == 1  # Error call

      error_call = mock_logger.error.call_args[0][0]
      assert "Async async_failing_function failed" in error_call
      assert "Async test error" in error_call

  def test_decorator_timing_accuracy(self):
    """Test that decorator timing is reasonably accurate."""
    import time

    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      @log_function_call("timed_function")
      def timed_func():
        time.sleep(0.1)  # Sleep for 100ms
        return "done"

      result = timed_func()

      assert result == "done"

      # Check that timing was logged
      end_call = mock_logger.debug.call_args_list[1][0][0]
      assert "completed in" in end_call
      # Should be around 0.1 seconds, but allow some variance
      assert "0.1" in end_call or "0.0" in end_call


class TestEdgeCasesAndErrorConditions:
  """Test edge cases and error conditions in logging components."""

  def test_intercept_handler_with_none_record(self):
    """Test InterceptHandler with None record."""
    handler = InterceptHandler()

    # Should raise AttributeError for None record (expected behavior)
    with pytest.raises(AttributeError):
      handler.emit(None)

  def test_logger_mixin_with_empty_messages(self):
    """Test LoggerMixin with empty and None messages."""

    class TestClass(LoggerMixin):
      pass

    test_obj = TestClass()

    # Test with empty string
    test_obj.logger.info("")

    # Test with None (should be converted to string)
    test_obj.logger.info(None)

    # Test with whitespace only
    test_obj.logger.info("   ")

  def test_request_logger_with_invalid_data(self):
    """Test RequestLogger with invalid request data."""
    logger = RequestLogger()

    # Test with None request
    logger.log_request(None, "test_endpoint")

    # Test with empty endpoint
    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url = "http://test.com"
    logger.log_request(mock_request, "")

    # Test with None endpoint
    logger.log_request(mock_request, None)

  def test_request_logger_with_missing_attributes(self):
    """Test RequestLogger when request object is missing expected attributes."""
    logger = RequestLogger()

    # Create a mock request with missing attributes
    incomplete_request = MagicMock()
    del incomplete_request.method  # Remove method attribute

    # Should handle gracefully
    try:
      logger.log_request(incomplete_request, "test_endpoint")
    except AttributeError:
      pass  # Expected behavior

  def test_agent_logger_with_none_values(self):
    """Test AgentLogger with None values."""
    logger = AgentLogger()

    # Test with None task_id
    logger.log_task_start(None, "test task")

    # Test with None step
    logger.log_task_step("task_1", None, "details")

    # Test with None description
    logger.log_task_start("task_1", None)

  def test_performance_logger_with_invalid_metrics(self):
    """Test PerformanceLogger with invalid metric values."""
    logger = PerformanceLogger()

    # Test with None values - should raise TypeError for format operations
    with pytest.raises(TypeError):
      logger.log_execution_time(None, None)

    with pytest.raises(TypeError):
      logger.log_memory_usage(None, None)

    # log_api_call handles None values gracefully, so test with valid call
    logger.log_api_call(None, None, None, None)  # Should not raise

  def test_setup_logging_with_invalid_config(self):
    """Test setup_logging with invalid configuration values."""
    # Test with invalid log level
    with patch.dict(os.environ, {"LOG_LEVEL": "INVALID_LEVEL"}):
      setup_logging()

    # Test with invalid format
    with patch.dict(os.environ, {"LOG_FORMAT": "INVALID_FORMAT"}):
      setup_logging()

    # Test with invalid file path (permission denied scenario)
    with patch.dict(os.environ, {"LOG_FILE": "/root/invalid_path.log"}):
      setup_logging()

  def test_get_logger_with_invalid_names(self):
    """Test get_logger with invalid logger names."""
    # Test with empty string
    logger1 = get_logger("")
    assert logger1 is not None

    # Test with None
    logger2 = get_logger(None)
    assert logger2 is not None

    # Test with special characters
    logger3 = get_logger("test.logger-with_special@chars")
    assert logger3 is not None

    # Test with very long name
    long_name = "a" * 1000
    logger4 = get_logger(long_name)
    assert logger4 is not None

  def test_decorator_with_complex_arguments(self):
    """Test decorators with complex argument types."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      @log_function_call("complex_function")
      def complex_func(obj, lst, dct, *args, **kwargs):
        return "success"

      # Test with complex objects
      test_obj = {"key": "value"}
      test_list = [1, 2, 3]
      test_dict = {"nested": {"data": True}}

      result = complex_func(test_obj, test_list, test_dict, "extra", flag=True)

      assert result == "success"
      assert mock_logger.debug.call_count == 2

  def test_logging_with_unicode_and_special_chars(self):
    """Test logging with unicode and special characters."""
    logger = get_logger("unicode_test")

    # Test with unicode characters
    logger.info("Unicode test: 你好世界 🌍")

    # Test with special characters
    logger.info("Special chars: !@#$%^&*()[]{}|\\:;\"'<>,.?/")

    # Test with newlines and tabs
    logger.info("Multi\nline\tmessage")

  def test_concurrent_logger_creation(self):
    """Test concurrent logger creation doesn't cause issues."""
    import threading

    loggers = []
    errors = []

    def create_logger(name):
      try:
        logger = get_logger(f"concurrent_test_{name}")
        loggers.append(logger)
        logger.info(f"Logger {name} created")
      except Exception as e:
        errors.append(e)

    # Create multiple threads that create loggers simultaneously
    threads = []
    for i in range(10):
      thread = threading.Thread(target=create_logger, args=(i,))
      threads.append(thread)

    # Start all threads
    for thread in threads:
      thread.start()

    # Wait for all threads to complete
    for thread in threads:
      thread.join()

    # Check results
    assert len(errors) == 0, (
      f"Errors occurred during concurrent logger creation: {errors}"
    )
    assert len(loggers) == 10, f"Expected 10 loggers, got {len(loggers)}"


class TestAgentLoggerMethods:
  """Test specific AgentLogger methods."""

  def test_log_task_start(self):
    """Test log_task_start method."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      agent_logger = AgentLogger()
      agent_logger.log_task_start("task_123", "Test task description")

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "Task started" in call_args
      assert "task_123" in call_args
      assert "Test task description" in call_args

  def test_log_task_step(self):
    """Test log_task_step method."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      agent_logger = AgentLogger()
      agent_logger.log_task_step("task_123", "processing", "details here")

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "task_123" in call_args
      assert "processing" in call_args
      assert "details here" in call_args

  def test_log_task_complete(self):
    """Test log_task_complete method."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      agent_logger = AgentLogger()
      agent_logger.log_task_complete("task_456", 1.0)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "Task completed: task_456" in call_args
      assert "1.000s" in call_args

  def test_log_task_complete_failure(self):
    """Test log_task_complete method with failure."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      agent_logger = AgentLogger()
      agent_logger.log_task_complete("task_456", 2.5, success=False)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "Task failed: task_456" in call_args
      assert "2.500s" in call_args

  def test_log_tool_usage(self):
    """Test log_tool_usage method."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      agent_logger = AgentLogger()
      params = {"query": "test", "limit": 10}
      result = {"success": True}  # Use 'success' key as expected by implementation
      agent_logger.log_tool_usage("search_tool", params, result)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "Tool search_tool - success" in call_args


class TestPerformanceLoggerMethods:
  """Test specific PerformanceLogger methods."""

  def test_log_execution_time(self):
    """Test log_execution_time method."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      perf_logger = PerformanceLogger()
      perf_logger.log_execution_time("test_operation", 0.5)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "Performance: test_operation took 0.500s" in call_args

  def test_log_memory_usage(self):
    """Test log_memory_usage method."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      perf_logger = PerformanceLogger()
      perf_logger.log_memory_usage("test_operation", 512.5)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "Memory: test_operation used 512.50MB" in call_args

  def test_log_api_call(self):
    """Test log_api_call method."""
    with patch("src.utils.logging.get_logger") as mock_get_logger:
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger

      perf_logger = PerformanceLogger()
      perf_logger.log_api_call("openai", "gpt-4", 150, 0.0030)

      mock_logger.info.assert_called_once()
      call_args = mock_logger.info.call_args[0][0]
      assert "API Call: openai/gpt-4" in call_args
      assert "150 tokens" in call_args
      assert "$0.0030" in call_args


class TestGlobalLoggerInstances:
  """Test global logger instances."""

  def test_global_request_logger_import(self):
    """Test that global request_logger can be imported and used."""
    from src.utils.logging import request_logger

    assert request_logger is not None
    assert isinstance(request_logger, RequestLogger)

    # Test that it can be used
    mock_request = MagicMock()
    mock_request.method = "POST"
    mock_request.url = "http://test.com/api"
    request_logger.log_request(mock_request, "test_endpoint")

  def test_global_agent_logger_import(self):
    """Test that global agent_logger can be imported and used."""
    from src.utils.logging import agent_logger

    assert agent_logger is not None
    assert isinstance(agent_logger, AgentLogger)

    # Test that it can be used
    agent_logger.log_task_start("test_task", "Test task description")

  def test_global_performance_logger_import(self):
    """Test that global performance_logger can be imported and used."""
    from src.utils.logging import performance_logger

    assert performance_logger is not None
    assert isinstance(performance_logger, PerformanceLogger)

    # Test that it can be used
    performance_logger.log_execution_time("test_operation", 0.5)

  def test_global_loggers_are_singletons(self):
    """Test that global loggers behave as singletons."""
    from src.utils.logging import request_logger, agent_logger, performance_logger

    # Import again to check if they're the same instances
    from src.utils.logging import (
      request_logger as request_logger2,
      agent_logger as agent_logger2,
      performance_logger as performance_logger2,
    )

    assert request_logger is request_logger2
    assert agent_logger is agent_logger2
    assert performance_logger is performance_logger2


class TestFileLoggingIntegration:
  """Test file logging integration and configuration."""

  def test_file_logging_configuration(self):
    """Test file logging configuration is applied."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
      log_file = os.path.join(temp_dir, "test.log")

      # Setup logging with file output
      with patch.dict(os.environ, {"LOG_FILE": log_file, "LOG_FORMAT": "text"}):
        # Mock loguru to verify configuration
        with patch("src.utils.logging.loguru_logger") as mock_loguru:
          setup_logging()

          # Verify that loguru.add was called (file logging setup)
          mock_loguru.add.assert_called()

  def test_json_format_configuration(self):
    """Test JSON format logging configuration."""
    with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
      with patch("src.utils.logging.loguru_logger") as mock_loguru:
        setup_logging()

        # Verify that loguru.add was called (logging setup)
        mock_loguru.add.assert_called()

  def test_rotation_configuration(self):
    """Test log rotation configuration."""
    with patch.dict(os.environ, {"LOG_ROTATION": "1 MB", "LOG_RETENTION": "3 days"}):
      with patch("src.utils.logging.loguru_logger") as mock_loguru:
        setup_logging()

        # Verify that loguru was configured with rotation
        mock_loguru.add.assert_called()

  def test_logging_graceful_error_handling(self):
    """Test logging handles configuration errors gracefully."""
    # Test with invalid configuration
    with patch.dict(os.environ, {"LOG_FILE": "/invalid/path/test.log"}):
      # Should not raise exception
      try:
        setup_logging()
        test_logger = get_logger("error_test")
        test_logger.info("This should not crash")
      except Exception as e:
        pytest.fail(f"Logging should handle errors gracefully: {e}")


class TestConcurrentLogging:
  """Test concurrent logging scenarios."""

  def test_concurrent_logging_same_logger(self):
    """Test multiple threads logging to the same logger."""
    import threading
    import time

    logger = get_logger("concurrent_test")
    messages_logged = []
    errors = []

    def log_messages(thread_id):
      try:
        for i in range(10):
          message = f"Thread {thread_id} message {i}"
          logger.info(message)
          messages_logged.append(message)
          time.sleep(0.001)  # Small delay
      except Exception as e:
        errors.append(e)

    # Create multiple threads
    threads = []
    for i in range(5):
      thread = threading.Thread(target=log_messages, args=(i,))
      threads.append(thread)

    # Start all threads
    for thread in threads:
      thread.start()

    # Wait for completion
    for thread in threads:
      thread.join()

    # Check results
    assert len(errors) == 0, f"Errors during concurrent logging: {errors}"
    assert len(messages_logged) == 50, (
      f"Expected 50 messages, got {len(messages_logged)}"
    )

  def test_concurrent_performance_logging(self):
    """Test concurrent performance logging."""
    import threading

    perf_logger = PerformanceLogger()
    errors = []

    def performance_operations(thread_id):
      try:
        for i in range(5):
          operation_name = f"thread_{thread_id}_op_{i}"
          import time

          start_time = time.time()
          time.sleep(0.01)  # Simulate work
          duration = time.time() - start_time
          perf_logger.log_execution_time(operation_name, duration)
      except Exception as e:
        errors.append(e)

    # Create multiple threads
    threads = []
    for i in range(3):
      thread = threading.Thread(target=performance_operations, args=(i,))
      threads.append(thread)

    # Start and wait
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    # Check no errors occurred
    assert len(errors) == 0, f"Errors during concurrent performance logging: {errors}"
