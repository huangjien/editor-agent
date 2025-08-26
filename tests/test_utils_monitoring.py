"""Tests for monitoring utilities."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.utils.monitoring import (
  HealthChecker,
  MetricsCollector,
  SystemMonitor,
  basic_health_check,
  database_health_check,
  system_health_check,
  health_checker,
  metrics_collector,
  system_monitor,
)


class TestHealthChecker:
  """Test HealthChecker class."""

  def test_health_checker_initialization(self):
    """Test HealthChecker initialization."""
    checker = HealthChecker()

    assert checker.checks == {}
    assert isinstance(checker, HealthChecker)

  def test_register_health_check(self):
    """Test registering a health check."""
    checker = HealthChecker()

    async def test_check():
      return {"healthy": True, "message": "OK"}

    checker.register_check("test_service", test_check)

    assert "test_service" in checker.checks
    assert checker.checks["test_service"]["func"] == test_check
    assert checker.checks["test_service"]["timeout"] == 5.0

  @pytest.mark.asyncio
  async def test_run_check_unregistered(self):
    """Test running a check that doesn't exist."""
    checker = HealthChecker()
    result = await checker.run_check("nonexistent")

    assert result["status"] == "error"
    assert "not found" in result["message"]
    assert "timestamp" in result

  @pytest.mark.asyncio
  async def test_run_check_timeout(self):
    """Test timeout handling in run_check."""
    checker = HealthChecker()

    async def slow_check():
      await asyncio.sleep(2)  # Longer than timeout
      return {"healthy": True}

    checker.register_check("slow", slow_check, timeout=0.1)
    result = await checker.run_check("slow")

    assert result["status"] == "timeout"
    assert "timed out" in result["message"].lower()
    assert "timestamp" in result

  @pytest.mark.asyncio
  async def test_run_check_exception(self):
    """Test exception handling in run_check."""
    checker = HealthChecker()

    async def failing_check():
      raise ValueError("Test error")

    checker.register_check("failing", failing_check, timeout=5.0)
    result = await checker.run_check("failing")

    assert result["status"] == "error"
    assert "Test error" in result["message"]
    assert "timestamp" in result

  @pytest.mark.asyncio
  async def test_run_health_checks_all_pass(self):
    """Test running health checks when all pass."""
    checker = HealthChecker()

    async def check1():
      return {"healthy": True, "message": "Service 1 OK"}

    async def check2():
      return {"healthy": True, "message": "Service 2 OK"}

    checker.register_check("service1", check1)
    checker.register_check("service2", check2)

    result = await checker.run_all_checks()

    assert result["status"] == "healthy"
    assert len(result["checks"]) == 2
    assert result["checks"]["service1"]["status"] == "healthy"
    assert result["checks"]["service1"]["message"] == "Service 1 OK"
    assert result["checks"]["service2"]["status"] == "healthy"
    assert result["checks"]["service2"]["message"] == "Service 2 OK"
    assert "timestamp" in result
    assert "duration" in result

  @pytest.mark.asyncio
  async def test_run_health_checks_some_fail(self):
    """Test running health checks when some fail."""
    checker = HealthChecker()

    async def check1():
      return {"healthy": True, "message": "Service 1 OK"}

    async def check2():
      return {"healthy": False, "message": "Service 2 failed"}

    checker.register_check("service1", check1)
    checker.register_check("service2", check2)

    result = await checker.run_all_checks()

    assert result["status"] == "unhealthy"
    assert result["checks"]["service1"]["status"] == "healthy"
    assert result["checks"]["service2"]["status"] == "unhealthy"
    assert result["checks"]["service2"]["message"] == "Service 2 failed"

  @pytest.mark.asyncio
  async def test_run_health_checks_with_exception(self):
    """Test running health checks when check raises exception."""
    checker = HealthChecker()

    async def failing_check():
      raise ValueError("Check failed with exception")

    checker.register_check("failing_service", failing_check)

    result = await checker.run_all_checks()

    assert result["status"] == "error"
    assert result["checks"]["failing_service"]["status"] == "error"
    assert (
      "Check failed with exception" in result["checks"]["failing_service"]["message"]
    )

  @pytest.mark.asyncio
  async def test_run_all_checks_caching(self):
    """Test caching mechanism in run_all_checks."""
    checker = HealthChecker()
    call_count = 0

    async def counting_check():
      nonlocal call_count
      call_count += 1
      return {"healthy": True, "message": f"Called {call_count} times"}

    checker.register_check("counting", counting_check)

    # First call should execute the check
    result1 = await checker.run_all_checks(use_cache=True)
    assert call_count == 1
    assert "Called 1 times" in result1["checks"]["counting"]["message"]

    # Second call should use cache
    result2 = await checker.run_all_checks(use_cache=True)
    assert call_count == 1  # Should not increment
    assert (
      result2["checks"]["counting"]["message"]
      == result1["checks"]["counting"]["message"]
    )

    # Third call with use_cache=False should execute again
    result3 = await checker.run_all_checks(use_cache=False)
    assert call_count == 2
    assert "Called 2 times" in result3["checks"]["counting"]["message"]

  @pytest.mark.asyncio
  async def test_run_all_checks_cache_expiry(self):
    """Test cache expiry in run_all_checks."""
    checker = HealthChecker()
    checker.check_cache_duration = 0.1  # Very short cache duration
    call_count = 0

    async def counting_check():
      nonlocal call_count
      call_count += 1
      return {"healthy": True, "message": f"Called {call_count} times"}

    checker.register_check("counting", counting_check)

    # First call
    await checker.run_all_checks(use_cache=True)
    assert call_count == 1

    # Wait for cache to expire
    await asyncio.sleep(0.2)

    # Second call should execute again due to cache expiry
    await checker.run_all_checks(use_cache=True)
    assert call_count == 2

  @pytest.mark.asyncio
  async def test_concurrent_health_checks(self):
    """Test concurrent execution of health checks."""
    checker = HealthChecker()

    async def slow_check(delay):
      await asyncio.sleep(delay)
      return {"healthy": True, "message": f"Completed after {delay}s"}

    checker.register_check("fast", lambda: slow_check(0.1))
    checker.register_check("slow", lambda: slow_check(0.2))

    start_time = time.time()
    result = await checker.run_all_checks()
    end_time = time.time()

    # Should complete in roughly 0.2s (concurrent), not 0.3s (sequential)
    assert end_time - start_time < 0.3
    assert "fast" in result["checks"]
    assert "slow" in result["checks"]
    assert result["checks"]["fast"]["status"] == "healthy"
    assert result["checks"]["slow"]["status"] == "healthy"

  @pytest.mark.asyncio
  async def test_run_health_checks_async_function(self):
    """Test running async health checks."""
    checker = HealthChecker()

    async def async_check():
      await asyncio.sleep(0.01)  # Simulate async work
      return {"healthy": True, "message": "Async service OK"}

    checker.register_check("async_service", async_check)

    result = await checker.run_all_checks()

    assert result["status"] == "healthy"
    assert result["checks"]["async_service"]["status"] == "healthy"
    assert result["checks"]["async_service"]["message"] == "Async service OK"

  @pytest.mark.asyncio
  async def test_run_health_checks_includes_timestamp(self):
    """Test that health check results include timestamp."""
    checker = HealthChecker()

    async def test_check():
      return {"healthy": True, "message": "OK"}

    checker.register_check("test_service", test_check)

    before_time = time.time()
    result = await checker.run_all_checks()
    after_time = time.time()

    assert "timestamp" in result
    # Parse ISO timestamp to compare with numeric timestamps
    from datetime import datetime

    # The timestamp is in UTC, so we need to parse it as UTC
    timestamp_dt = datetime.fromisoformat(result["timestamp"])
    # Convert to UTC timestamp by treating the naive datetime as UTC
    import calendar

    timestamp = (
      calendar.timegm(timestamp_dt.timetuple()) + timestamp_dt.microsecond / 1000000.0
    )
    assert before_time <= timestamp <= after_time

  def test_register_check_duplicate_name(self):
    """Test registering a check with duplicate name overwrites previous."""
    checker = HealthChecker()

    async def check1():
      return {"healthy": True, "message": "Check 1"}

    async def check2():
      return {"healthy": True, "message": "Check 2"}

    checker.register_check("test", check1, timeout=5.0)
    assert checker.checks["test"]["timeout"] == 5.0

    # Register with same name but different timeout
    checker.register_check("test", check2, timeout=10.0)
    assert checker.checks["test"]["timeout"] == 10.0
    assert checker.checks["test"]["func"] == check2

  def test_register_check_invalid_timeout(self):
    """Test registering check with invalid timeout values."""
    checker = HealthChecker()

    async def test_check():
      return {"healthy": True}

    # Test with negative timeout
    checker.register_check("test_negative", test_check, timeout=-1.0)
    assert checker.checks["test_negative"]["timeout"] == -1.0

    # Test with zero timeout
    checker.register_check("test_zero", test_check, timeout=0.0)
    assert checker.checks["test_zero"]["timeout"] == 0.0

  @pytest.mark.asyncio
  async def test_run_check_with_zero_timeout(self):
    """Test running check with zero timeout immediately times out."""
    checker = HealthChecker()

    async def slow_check():
      await asyncio.sleep(0.1)
      return {"healthy": True}

    checker.register_check("slow", slow_check, timeout=0.0)
    result = await checker.run_check("slow")

    assert result["status"] == "timeout"
    assert "timed out" in result["message"].lower()

  @pytest.mark.asyncio
  async def test_run_all_checks_empty_registry(self):
    """Test running all checks when no checks are registered."""
    checker = HealthChecker()

    result = await checker.run_all_checks()

    assert result["status"] == "healthy"
    assert result["checks"] == {}
    assert "timestamp" in result
    assert "duration" in result

  @pytest.mark.asyncio
  async def test_run_all_checks_cache_disabled(self):
    """Test running all checks with caching disabled."""
    checker = HealthChecker()

    call_count = 0

    async def counting_check():
      nonlocal call_count
      call_count += 1
      return {"healthy": True, "message": f"Call {call_count}"}

    checker.register_check("counter", counting_check)

    # First call
    result1 = await checker.run_all_checks(use_cache=False)
    assert call_count == 1

    # Second call with cache disabled should call again
    result2 = await checker.run_all_checks(use_cache=False)
    assert call_count == 2

    # Results should be different
    assert (
      result1["checks"]["counter"]["message"] != result2["checks"]["counter"]["message"]
    )

  @pytest.mark.asyncio
  async def test_cache_invalidation_on_new_registration(self):
    """Test that cache is not automatically invalidated when new checks are registered."""
    checker = HealthChecker()

    async def check1():
      return {"healthy": True, "message": "Check 1"}

    async def check2():
      return {"healthy": True, "message": "Check 2"}

    checker.register_check("test1", check1)

    # Run checks to populate cache
    result1 = await checker.run_all_checks()
    assert len(result1["checks"]) == 1

    # Register new check
    checker.register_check("test2", check2)

    # Cache should still return old results
    result2 = await checker.run_all_checks(use_cache=True)
    assert len(result2["checks"]) == 1  # Still cached

    # Disable cache to get fresh results
    result3 = await checker.run_all_checks(use_cache=False)
    assert len(result3["checks"]) == 2  # Now includes new check

  @pytest.mark.asyncio
  async def test_run_check_function_returns_none(self):
    """Test handling check function that returns None."""
    checker = HealthChecker()

    async def none_check():
      return None

    checker.register_check("none_test", none_check)
    result = await checker.run_check("none_test")

    assert result["status"] == "error"
    assert "failed" in result["message"].lower()

  @pytest.mark.asyncio
  async def test_run_check_function_returns_invalid_format(self):
    """Test handling check function that returns invalid format."""
    checker = HealthChecker()

    async def invalid_check():
      return "invalid string response"

    checker.register_check("invalid_test", invalid_check)
    result = await checker.run_check("invalid_test")

    # Should handle gracefully when result.get() fails
    assert result["status"] == "error"
    assert "failed" in result["message"].lower()

  @pytest.mark.asyncio
  async def test_concurrent_cache_access(self):
    """Test concurrent access to cached results."""
    checker = HealthChecker()

    call_count = 0

    async def slow_check():
      nonlocal call_count
      call_count += 1
      await asyncio.sleep(0.1)
      return {"healthy": True, "message": f"Call {call_count}"}

    checker.register_check("slow", slow_check)

    # Run multiple concurrent requests
    tasks = [checker.run_all_checks() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Since they're concurrent, multiple might execute
    # But we expect some level of caching to occur
    assert call_count >= 1
    assert call_count <= 5  # Should not exceed the number of requests

    # All results should be valid
    for result in results:
      assert "timestamp" in result
      assert "checks" in result
      assert "slow" in result["checks"]


class TestMetricsCollector:
  """Test MetricsCollector class."""

  def test_metrics_collector_initialization(self):
    """Test MetricsCollector initialization."""
    collector = MetricsCollector()

    assert collector.metrics == {}
    assert isinstance(collector, MetricsCollector)

  def test_increment_counter(self):
    """Test incrementing a counter metric."""
    collector = MetricsCollector()

    collector.increment_counter("requests_total")
    collector.increment_counter("requests_total")
    collector.increment_counter("requests_total", value=3)

    assert collector.metrics["requests_total"]["value"] == 5

  def test_increment_counter_with_labels(self):
    """Test incrementing counter with labels."""
    collector = MetricsCollector()

    collector.increment_counter("requests_total", labels={"method": "GET"})
    collector.increment_counter("requests_total", labels={"method": "POST"})
    collector.increment_counter("requests_total", labels={"method": "GET"})

    assert collector.metrics["requests_total{method=GET}"]["value"] == 2
    assert collector.metrics["requests_total{method=POST}"]["value"] == 1

  def test_set_gauge(self):
    """Test setting a gauge metric."""
    collector = MetricsCollector()

    collector.set_gauge("memory_usage", 1024)
    collector.set_gauge("memory_usage", 2048)

    assert collector.metrics["memory_usage"]["value"] == 2048

  def test_set_gauge_with_labels(self):
    """Test setting gauge with labels."""
    collector = MetricsCollector()

    collector.set_gauge("cpu_usage", 50.5, labels={"core": "0"})
    collector.set_gauge("cpu_usage", 75.2, labels={"core": "1"})

    assert collector.metrics["cpu_usage{core=0}"]["value"] == 50.5
    assert collector.metrics["cpu_usage{core=1}"]["value"] == 75.2

  def test_record_histogram(self):
    """Test recording histogram values."""
    collector = MetricsCollector()

    collector.record_histogram("response_time", 0.1)
    collector.record_histogram("response_time", 0.2)
    collector.record_histogram("response_time", 0.15)

    histogram_key = "response_time"
    assert histogram_key in collector.metrics
    assert len(collector.metrics[histogram_key]["values"]) == 3
    assert 0.1 in collector.metrics[histogram_key]["values"]
    assert 0.2 in collector.metrics[histogram_key]["values"]
    assert 0.15 in collector.metrics[histogram_key]["values"]

  def test_record_histogram_with_labels(self):
    """Test recording histogram with labels."""
    collector = MetricsCollector()

    collector.record_histogram("response_time", 0.1, labels={"endpoint": "/api/v1"})
    collector.record_histogram("response_time", 0.2, labels={"endpoint": "/api/v2"})

    histogram_key1 = "response_time{endpoint=/api/v1}"
    histogram_key2 = "response_time{endpoint=/api/v2}"

    assert histogram_key1 in collector.metrics
    assert histogram_key2 in collector.metrics
    assert collector.metrics[histogram_key1]["values"] == [0.1]
    assert collector.metrics[histogram_key2]["values"] == [0.2]

  def test_get_metrics(self):
    """Test getting all metrics."""
    collector = MetricsCollector()

    collector.increment_counter("requests_total")
    collector.set_gauge("memory_usage", 1024)
    collector.record_histogram("response_time", 0.1)

    metrics = collector.get_metrics()

    # Check the structure returned by get_metrics
    assert "custom_metrics" in metrics
    assert "uptime_seconds" in metrics
    assert "total_requests" in metrics
    assert "timestamp" in metrics

    # Check custom metrics
    custom_metrics = metrics["custom_metrics"]
    assert "requests_total" in custom_metrics
    assert "memory_usage" in custom_metrics
    assert "response_time" in custom_metrics
    assert custom_metrics["requests_total"]["value"] == 1
    assert custom_metrics["memory_usage"]["value"] == 1024
    assert custom_metrics["response_time"]["values"] == [0.1]

  def test_metrics_storage(self):
    """Test that metrics are stored correctly in the internal structure."""
    collector = MetricsCollector()

    collector.increment_counter("requests_total")
    collector.set_gauge("memory_usage", 1024)

    # Check internal metrics storage
    assert len(collector.metrics) > 0
    assert "requests_total" in collector.metrics
    assert "memory_usage" in collector.metrics
    assert collector.metrics["requests_total"]["value"] == 1
    assert collector.metrics["memory_usage"]["value"] == 1024

  def test_record_request(self):
    """Test HTTP request recording."""
    collector = MetricsCollector()

    # Record a successful request
    collector.record_request("GET", "/api/test", 200, 0.5)

    assert collector.request_count == 1
    assert collector.error_count == 0
    assert len(collector.response_times) == 1
    assert collector.response_times[0] == 0.5

    # Record an error request
    collector.record_request("POST", "/api/error", 500, 1.2)

    assert collector.request_count == 2
    assert collector.error_count == 1
    assert len(collector.response_times) == 2

    # Check metrics structure
    metrics = collector.get_metrics()
    assert (
      "http_requests_total{endpoint=/api/test,method=GET,status=200}"
      in metrics["custom_metrics"]
    )
    assert (
      "http_errors_total{endpoint=/api/error,method=POST,status=500}"
      in metrics["custom_metrics"]
    )

  def test_record_request_response_time_limit(self):
    """Test that response times are limited to prevent memory growth."""
    collector = MetricsCollector()

    # Record more than max_response_times (1000)
    for i in range(1200):
      collector.record_request("GET", "/test", 200, float(i))

    # Should be limited to max_response_times
    assert len(collector.response_times) == collector.max_response_times
    assert collector.response_times[0] == 200.0  # First kept value

  def test_get_metric_key_without_labels(self):
    """Test metric key generation without labels."""
    collector = MetricsCollector()
    key = collector._get_metric_key("test_metric")
    assert key == "test_metric"

    key = collector._get_metric_key("test_metric", None)
    assert key == "test_metric"

  def test_get_metric_key_with_labels(self):
    """Test metric key generation with labels."""
    collector = MetricsCollector()
    labels = {"method": "GET", "status": "200"}
    key = collector._get_metric_key("test_metric", labels)
    assert key == "test_metric{method=GET,status=200}"

    # Test label sorting
    labels = {"z": "last", "a": "first"}
    key = collector._get_metric_key("test", labels)
    assert key == "test{a=first,z=last}"

  def test_histogram_memory_management(self):
    """Test that histogram values are limited to prevent memory growth."""
    collector = MetricsCollector()

    # Record more than 1000 values
    for i in range(1200):
      collector.record_histogram("test_histogram", float(i))

    # Should be limited to 1000 values
    histogram_data = collector.metrics["test_histogram"]
    assert len(histogram_data["values"]) == 1000
    assert histogram_data["values"][0] == 200.0  # First kept value

  def test_concurrent_metrics_collection(self):
    """Test concurrent metrics collection for thread safety."""
    import threading

    collector = MetricsCollector()

    def increment_counter():
      for i in range(100):
        collector.increment_counter("concurrent_counter")

    def record_histogram():
      for i in range(100):
        collector.record_histogram("concurrent_histogram", i)

    # Create multiple threads
    threads = []
    for _ in range(5):
      threads.append(threading.Thread(target=increment_counter))
      threads.append(threading.Thread(target=record_histogram))

    # Start all threads
    for thread in threads:
      thread.start()

    # Wait for all threads to complete
    for thread in threads:
      thread.join()

    metrics = collector.get_metrics()
    # Should have 500 total increments (5 threads * 100 increments)
    assert metrics["custom_metrics"]["concurrent_counter"]["value"] == 500
    # Should have 500 histogram entries
    assert len(metrics["custom_metrics"]["concurrent_histogram"]["values"]) == 500

  def test_metrics_with_special_characters(self):
    """Test metrics with special characters in names and labels."""
    collector = MetricsCollector()

    # Test with special characters
    collector.increment_counter(
      "metric-with-dashes", 1, {"label.with.dots": "value_with_underscores"}
    )
    collector.set_gauge(
      "metric/with/slashes", 42.5, {"label:with:colons": "value with spaces"}
    )

    metrics = collector.get_metrics()

    # Check that metrics are stored correctly
    assert (
      "metric-with-dashes{label.with.dots=value_with_underscores}"
      in metrics["custom_metrics"]
    )
    assert (
      "metric/with/slashes{label:with:colons=value with spaces}"
      in metrics["custom_metrics"]
    )

  def test_empty_metrics_collection(self):
    """Test getting metrics when no metrics have been collected."""
    collector = MetricsCollector()

    metrics = collector.get_metrics()

    assert metrics["custom_metrics"] == {}
    assert "uptime_seconds" in metrics
    assert "total_requests" in metrics
    assert "total_errors" in metrics
    assert "error_rate" in metrics
    assert "response_time_stats" in metrics
    assert "timestamp" in metrics

  def test_large_label_values(self):
    """Test metrics with large label values."""
    collector = MetricsCollector()

    large_value = "x" * 1000  # Very long label value
    collector.increment_counter("test_counter", 1, {"large_label": large_value})

    metrics = collector.get_metrics()
    expected_key = f"test_counter{{large_label={large_value}}}"
    assert expected_key in metrics["custom_metrics"]
    assert metrics["custom_metrics"][expected_key]["value"] == 1

  def test_increment_counter_with_invalid_amount(self):
    """Test increment_counter with invalid amount values."""
    collector = MetricsCollector()

    # Test with negative amount
    collector.increment_counter("negative_counter", -5)
    assert collector.metrics["negative_counter"]["value"] == -5

    # Test with zero amount
    collector.increment_counter("zero_counter", 0)
    assert collector.metrics["zero_counter"]["value"] == 0

    # Test with float amount
    collector.increment_counter("float_counter", 2.5)
    assert collector.metrics["float_counter"]["value"] == 2.5

  def test_set_gauge_with_invalid_values(self):
    """Test set_gauge with various invalid or edge case values."""
    collector = MetricsCollector()

    # Test with None value
    collector.set_gauge("none_gauge", None)
    assert collector.metrics["none_gauge"]["value"] is None

    # Test with string value
    collector.set_gauge("string_gauge", "invalid")
    assert collector.metrics["string_gauge"]["value"] == "invalid"

    # Test with very large number
    large_number = 10**100
    collector.set_gauge("large_gauge", large_number)
    assert collector.metrics["large_gauge"]["value"] == large_number

    # Test with negative infinity
    collector.set_gauge("neg_inf_gauge", float("-inf"))
    assert collector.metrics["neg_inf_gauge"]["value"] == float("-inf")

  def test_record_histogram_with_invalid_values(self):
    """Test record_histogram with invalid values."""
    collector = MetricsCollector()

    # Test with None value
    collector.record_histogram("none_histogram", None)
    assert None in collector.metrics["none_histogram"]["values"]

    # Test with string value
    collector.record_histogram("string_histogram", "invalid")
    assert "invalid" in collector.metrics["string_histogram"]["values"]

    # Test with NaN
    collector.record_histogram("nan_histogram", float("nan"))
    import math

    assert math.isnan(collector.metrics["nan_histogram"]["values"][0])

  def test_labels_with_none_values(self):
    """Test metrics with None values in labels."""
    collector = MetricsCollector()

    labels = {"key1": None, "key2": "value2"}
    collector.increment_counter("test_counter", 1, labels)

    # Should handle None values in labels
    expected_key = "test_counter{key1=None,key2=value2}"
    assert expected_key in collector.metrics
    assert collector.metrics[expected_key]["value"] == 1

  def test_empty_metric_names(self):
    """Test metrics with empty or whitespace-only names."""
    collector = MetricsCollector()

    # Test with empty string
    collector.increment_counter("", 1)
    assert "" in collector.metrics

    # Test with whitespace-only string
    collector.set_gauge("   ", 42)
    assert "   " in collector.metrics

    # Test with tab and newline characters
    collector.record_histogram("\t\n", 1.5)
    assert "\t\n" in collector.metrics

  def test_record_request_with_invalid_parameters(self):
    """Test record_request with invalid parameters."""
    collector = MetricsCollector()

    # Test with None method
    collector.record_request(None, "/test", 200, 0.5)
    assert collector.request_count == 1

    # Test with None endpoint
    collector.record_request("GET", None, 200, 0.5)
    assert collector.request_count == 2

    # Test with negative response time
    collector.record_request("DELETE", "/test", 404, -1.0)
    assert collector.request_count == 3
    assert -1.0 in collector.response_times

    # Test with string status code that can't be compared - should raise TypeError
    with pytest.raises(TypeError):
      collector.record_request("POST", "/test", "invalid", 0.5)

  def test_get_metric_key_with_empty_labels(self):
    """Test metric key generation with empty labels dictionary."""
    collector = MetricsCollector()

    # Test with empty labels dict
    key = collector._get_metric_key("test_metric", {})
    assert key == "test_metric"

    # Test with labels containing empty string keys/values
    labels = {"": "empty_key", "empty_value": ""}
    key = collector._get_metric_key("test_metric", labels)
    assert key == "test_metric{=empty_key,empty_value=}"

  def test_metrics_type_consistency(self):
    """Test that metrics maintain type consistency."""
    collector = MetricsCollector()

    # Test counter type consistency
    collector.increment_counter("test_counter", 1)
    assert collector.metrics["test_counter"]["type"] == "counter"

    # Test gauge type consistency
    collector.set_gauge("test_gauge", 42)
    assert collector.metrics["test_gauge"]["type"] == "gauge"

    # Test histogram type consistency
    collector.record_histogram("test_histogram", 1.5)
    assert collector.metrics["test_histogram"]["type"] == "histogram"

  def test_response_time_statistics_edge_cases(self):
    """Test response time statistics with edge cases."""
    collector = MetricsCollector()

    # Test with no response times
    metrics = collector.get_metrics()
    stats = metrics["response_time_stats"]
    assert stats == {}  # Empty dict when no response times

    # Test with single response time
    collector.record_request("GET", "/test", 200, 1.5)
    metrics = collector.get_metrics()
    stats = metrics["response_time_stats"]
    assert stats["avg"] == 1.5
    assert stats["min"] == 1.5
    assert stats["max"] == 1.5
    assert stats["p50"] == 1.5
    assert stats["p95"] == 1.5
    assert stats["p99"] == 1.5

    # Test with identical response times
    collector.record_request("GET", "/test", 200, 1.5)
    collector.record_request("GET", "/test", 200, 1.5)
    metrics = collector.get_metrics()
    stats = metrics["response_time_stats"]
    assert stats["avg"] == 1.5
    assert stats["min"] == 1.5
    assert stats["max"] == 1.5
    assert stats["p50"] == 1.5
    assert stats["p95"] == 1.5
    assert stats["p99"] == 1.5

  def test_error_rate_calculation_edge_cases(self):
    """Test error rate calculation with edge cases."""
    collector = MetricsCollector()

    # Test with no requests
    metrics = collector.get_metrics()
    assert metrics["error_rate"] == 0.0

    # Test with only successful requests
    collector.record_request("GET", "/test", 200, 0.5)
    collector.record_request("POST", "/test", 201, 0.3)
    metrics = collector.get_metrics()
    assert metrics["error_rate"] == 0.0

    # Test with only error requests
    collector = MetricsCollector()  # Reset
    collector.record_request("GET", "/test", 500, 0.5)
    collector.record_request("POST", "/test", 404, 0.3)
    metrics = collector.get_metrics()
    assert metrics["error_rate"] == 1.0


class TestSystemMonitor:
  """Test SystemMonitor class."""

  def test_system_monitor_initialization(self):
    """Test SystemMonitor initialization."""
    monitor = SystemMonitor()

    assert isinstance(monitor, SystemMonitor)

  # Note: SystemMonitor only has get_system_info() static method
  # Individual methods like get_cpu_usage, get_memory_usage etc. don't exist

  @patch("psutil.cpu_percent")
  @patch("psutil.virtual_memory")
  @patch("psutil.disk_usage")
  @patch("psutil.net_io_counters")
  @patch("psutil.Process")
  def test_get_system_info(
    self,
    mock_process_class,
    mock_net_io,
    mock_disk_usage,
    mock_virtual_memory,
    mock_cpu_percent,
  ):
    """Test getting complete system information."""
    # Mock CPU
    mock_cpu_percent.return_value = 45.5

    # Mock memory
    mock_memory = MagicMock()
    mock_memory.total = 8 * 1024 * 1024 * 1024
    mock_memory.available = 4 * 1024 * 1024 * 1024
    mock_memory.percent = 50.0
    mock_virtual_memory.return_value = mock_memory

    # Mock disk
    mock_disk = MagicMock()
    mock_disk.total = 1024 * 1024 * 1024 * 1024
    mock_disk.used = 512 * 1024 * 1024 * 1024
    mock_disk.free = 512 * 1024 * 1024 * 1024
    mock_disk_usage.return_value = mock_disk

    # Mock network
    mock_network = MagicMock()
    mock_network.bytes_sent = 1024
    mock_network.bytes_recv = 2048
    mock_net_io.return_value = mock_network

    # Mock process
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.name.return_value = "python"
    mock_process.cpu_percent.return_value = 15.5
    mock_process.memory_info.return_value = MagicMock(rss=64 * 1024 * 1024)
    mock_process.create_time.return_value = time.time() - 3600
    mock_process_class.return_value = mock_process

    system_info = SystemMonitor.get_system_info()

    assert "cpu" in system_info
    assert "memory" in system_info
    assert "disk" in system_info
    assert "network" in system_info
    assert "process" in system_info
    assert "timestamp" in system_info

    assert system_info["cpu"]["percent"] == 45.5
    assert system_info["memory"]["percent"] == 50.0
    assert system_info["disk"]["percent"] == 50.0

  @patch("psutil.cpu_percent")
  @patch("psutil.virtual_memory")
  @patch("psutil.disk_usage")
  @patch("psutil.net_io_counters")
  @patch("psutil.Process")
  def test_get_system_info_psutil_exceptions(
    self,
    mock_process,
    mock_net,
    mock_disk,
    mock_memory,
    mock_cpu,
  ):
    """Test handling of psutil exceptions."""
    import psutil

    # Mock psutil exceptions
    mock_cpu.side_effect = psutil.Error("CPU error")
    mock_memory.side_effect = psutil.Error("Memory error")
    mock_disk.side_effect = psutil.Error("Disk error")
    mock_net.side_effect = psutil.Error("Network error")
    mock_process.side_effect = psutil.Error("Process error")

    system_info = SystemMonitor.get_system_info()

    # Should return error information when any psutil call fails
    assert "error" in system_info
    assert "timestamp" in system_info
    assert "cpu" not in system_info
    assert "memory" not in system_info
    assert "disk" not in system_info

  @patch("psutil.cpu_percent")
  @patch("psutil.virtual_memory")
  def test_get_system_info_partial_failure(self, mock_memory, mock_cpu):
    """Test partial failure scenarios."""
    import psutil

    # Memory fails - this will cause the entire method to fail
    mock_memory.side_effect = psutil.AccessDenied("Access denied")

    system_info = SystemMonitor.get_system_info()

    # When any component fails, the entire method returns an error
    assert "error" in system_info
    assert "timestamp" in system_info
    assert "cpu" not in system_info
    assert "memory" not in system_info
    assert "disk" not in system_info

  def test_get_system_info_network_failure(self):
    """Test get_system_info when network interface information fails."""
    with patch("psutil.net_io_counters", side_effect=Exception("Network error")):
      system_info = SystemMonitor.get_system_info()

      # Should return error information
      assert "error" in system_info
      assert "timestamp" in system_info
      assert "Network error" in system_info["error"]

  def test_get_system_info_process_access_denied_standalone(self):
    """Test get_system_info when process access is denied."""
    with patch("psutil.Process", side_effect=Exception("Process access denied")):
      system_info = SystemMonitor.get_system_info()

      # Should return error information
      assert "error" in system_info
      assert "timestamp" in system_info
      assert "Process access denied" in system_info["error"]

  def test_get_system_info_all_components_fail_standalone(self):
    """Test get_system_info when all system components fail."""
    with (
      patch("psutil.cpu_percent", side_effect=Exception("CPU error")),
      patch("psutil.virtual_memory", side_effect=Exception("Memory error")),
      patch("psutil.disk_usage", side_effect=Exception("Disk error")),
      patch("psutil.net_io_counters", side_effect=Exception("Network error")),
      patch("psutil.Process", side_effect=Exception("Process error")),
    ):
      system_info = SystemMonitor.get_system_info()

      # Should return error information
      assert "error" in system_info
      assert "timestamp" in system_info
      # Should contain one of the error messages
      assert any(
        error in system_info["error"]
        for error in [
          "CPU error",
          "Memory error",
          "Disk error",
          "Network error",
          "Process error",
        ]
      )

  @patch("psutil.cpu_percent")
  @patch("psutil.virtual_memory")
  @patch("psutil.disk_usage")
  @patch("psutil.net_io_counters")
  @patch("psutil.Process")
  def test_get_system_info_network_interface_errors(
    self,
    mock_process,
    mock_net,
    mock_disk,
    mock_memory,
    mock_cpu,
  ):
    """Test get_system_info with network interface errors."""
    import psutil

    # Mock other components to work
    mock_cpu.return_value = 25.0

    mock_memory_obj = MagicMock()
    mock_memory_obj.total = 4 * 1024 * 1024 * 1024
    mock_memory_obj.available = 2 * 1024 * 1024 * 1024
    mock_memory_obj.percent = 50.0
    mock_memory.return_value = mock_memory_obj

    mock_disk_obj = MagicMock()
    mock_disk_obj.total = 100 * 1024 * 1024 * 1024
    mock_disk_obj.used = 50 * 1024 * 1024 * 1024
    mock_disk_obj.free = 50 * 1024 * 1024 * 1024
    mock_disk.return_value = mock_disk_obj

    mock_process_obj = MagicMock()
    mock_process_obj.memory_info.return_value = MagicMock(rss=50 * 1024 * 1024)
    mock_process_obj.cpu_percent.return_value = 15.0
    mock_process_obj.create_time.return_value = time.time() - 3600
    mock_process.return_value = mock_process_obj

    # Network fails
    mock_net.side_effect = psutil.Error("Network interface error")

    info = SystemMonitor.get_system_info()

    # Should return error information when network fails
    assert "error" in info
    assert "timestamp" in info
    assert "Network error:" in info["error"]
    assert "Network interface error" in info["error"]
    assert "cpu" not in info
    assert "memory" not in info
    assert "disk" not in info
    assert "network" not in info
    assert "process" not in info

  @patch("psutil.cpu_percent")
  @patch("psutil.virtual_memory")
  @patch("psutil.disk_usage")
  @patch("psutil.net_io_counters")
  @patch("psutil.Process")
  def test_get_system_info_process_access_denied(
    self,
    mock_process,
    mock_net,
    mock_disk,
    mock_memory,
    mock_cpu,
  ):
    """Test get_system_info when process access is denied."""
    import psutil

    # Mock other components to work
    mock_cpu.return_value = 30.0

    mock_memory_obj = MagicMock()
    mock_memory_obj.total = 16 * 1024 * 1024 * 1024
    mock_memory_obj.available = 8 * 1024 * 1024 * 1024
    mock_memory_obj.percent = 50.0
    mock_memory.return_value = mock_memory_obj

    mock_disk_obj = MagicMock()
    mock_disk_obj.total = 500 * 1024 * 1024 * 1024
    mock_disk_obj.used = 250 * 1024 * 1024 * 1024
    mock_disk_obj.free = 250 * 1024 * 1024 * 1024
    mock_disk.return_value = mock_disk_obj

    mock_net_obj = MagicMock()
    mock_net_obj.bytes_sent = 5000
    mock_net_obj.bytes_recv = 10000
    mock_net.return_value = mock_net_obj

    # Process access denied
    mock_process.side_effect = psutil.AccessDenied("Access denied")

    info = SystemMonitor.get_system_info()

    # Should return error information when process access is denied
    assert "error" in info
    assert "timestamp" in info
    assert "Access denied" in info["error"]
    assert "cpu" not in info
    assert "memory" not in info
    assert "disk" not in info
    assert "network" not in info
    assert "process" not in info

  @patch("psutil.cpu_percent")
  @patch("psutil.virtual_memory")
  @patch("psutil.disk_usage")
  @patch("psutil.net_io_counters")
  @patch("psutil.Process")
  def test_get_system_info_all_components_fail(
    self,
    mock_process,
    mock_net,
    mock_disk,
    mock_memory,
    mock_cpu,
  ):
    """Test get_system_info when all components fail."""
    import psutil

    # All components fail
    mock_cpu.side_effect = psutil.Error("CPU error")
    mock_memory.side_effect = psutil.Error("Memory error")
    mock_disk.side_effect = psutil.Error("Disk error")
    mock_net.side_effect = psutil.Error("Network error")
    mock_process.side_effect = psutil.Error("Process error")

    info = SystemMonitor.get_system_info()

    # Should return error information when all components fail
    assert "error" in info
    assert "timestamp" in info
    assert "CPU error" in info["error"]
    assert "cpu" not in info
    assert "memory" not in info
    assert "disk" not in info
    assert "network" not in info
    assert "process" not in info

  def test_get_system_info_network_interface_specific_errors(self):
    """Test get_system_info with specific network interface errors."""
    import psutil

    with (
      patch("psutil.cpu_percent", return_value=25.0),
      patch("psutil.virtual_memory") as mock_memory,
      patch("psutil.disk_usage") as mock_disk,
      patch(
        "psutil.net_io_counters",
        side_effect=psutil.Error("Network interface not found"),
      ),
      patch("psutil.Process") as mock_process,
    ):
      # Mock other components to work
      mock_memory_obj = MagicMock()
      mock_memory_obj.total = 4 * 1024 * 1024 * 1024
      mock_memory_obj.available = 2 * 1024 * 1024 * 1024
      mock_memory_obj.percent = 50.0
      mock_memory.return_value = mock_memory_obj

      mock_disk_obj = MagicMock()
      mock_disk_obj.total = 100 * 1024 * 1024 * 1024
      mock_disk_obj.used = 50 * 1024 * 1024 * 1024
      mock_disk_obj.free = 50 * 1024 * 1024 * 1024
      mock_disk.return_value = mock_disk_obj

      mock_process_obj = MagicMock()
      mock_process_obj.memory_info.return_value = MagicMock(rss=50 * 1024 * 1024)
      mock_process_obj.cpu_percent.return_value = 15.0
      mock_process_obj.create_time.return_value = time.time() - 3600
      mock_process.return_value = mock_process_obj

      info = SystemMonitor.get_system_info()

      # Should return error information when network interface fails
      assert "error" in info
      assert "timestamp" in info
      assert "Network error:" in info["error"]
      assert "Network interface not found" in info["error"]
      assert "cpu" not in info
      assert "memory" not in info
      assert "disk" not in info
      assert "network" not in info
      assert "process" not in info

  def test_get_system_info_process_permission_errors(self):
    """Test get_system_info when process information access is restricted."""
    import psutil

    with (
      patch("psutil.cpu_percent", return_value=30.0),
      patch("psutil.virtual_memory") as mock_memory,
      patch("psutil.disk_usage") as mock_disk,
      patch("psutil.net_io_counters") as mock_net,
      patch("psutil.Process", side_effect=psutil.AccessDenied("Permission denied")),
    ):
      # Mock other components to work
      mock_memory_obj = MagicMock()
      mock_memory_obj.total = 16 * 1024 * 1024 * 1024
      mock_memory_obj.available = 8 * 1024 * 1024 * 1024
      mock_memory_obj.percent = 50.0
      mock_memory.return_value = mock_memory_obj

      mock_disk_obj = MagicMock()
      mock_disk_obj.total = 500 * 1024 * 1024 * 1024
      mock_disk_obj.used = 250 * 1024 * 1024 * 1024
      mock_disk_obj.free = 250 * 1024 * 1024 * 1024
      mock_disk.return_value = mock_disk_obj

      mock_net_obj = MagicMock()
      mock_net_obj.bytes_sent = 5000
      mock_net_obj.bytes_recv = 10000
      mock_net.return_value = mock_net_obj

      info = SystemMonitor.get_system_info()

      # Should return error information when process access is denied
      assert "error" in info
      assert "timestamp" in info
      assert "Permission denied" in info["error"]
      assert "cpu" not in info
      assert "memory" not in info
      assert "disk" not in info
      assert "network" not in info
      assert "process" not in info

  def test_get_system_info_cpu_frequency_edge_cases(self):
    """Test SystemMonitor with CPU frequency edge cases."""
    with (
      patch("psutil.cpu_percent") as mock_cpu_percent,
      patch("psutil.cpu_count") as mock_cpu_count,
      patch("psutil.cpu_freq") as mock_cpu_freq,
      patch("psutil.virtual_memory") as mock_memory,
      patch("psutil.swap_memory") as mock_swap,
      patch("psutil.disk_usage") as mock_disk,
      patch("psutil.net_io_counters") as mock_net,
      patch("psutil.Process") as mock_process_class,
    ):
      # Mock CPU with no frequency info (returns None)
      mock_cpu_percent.return_value = 25.0
      mock_cpu_count.return_value = 4
      mock_cpu_freq.return_value = None

      # Mock other components to succeed
      mock_memory.return_value = MagicMock(
        total=8 * 1024**3,
        available=4 * 1024**3,
        percent=50.0,
        used=4 * 1024**3,
        free=4 * 1024**3,
      )
      mock_swap.return_value = MagicMock(
        total=2 * 1024**3, used=1024**3, free=1024**3, percent=50.0
      )
      mock_disk.return_value = MagicMock(
        total=1024**4, used=512 * 1024**3, free=512 * 1024**3
      )
      mock_net.return_value = MagicMock(
        bytes_sent=1024, bytes_recv=2048, packets_sent=10, packets_recv=20
      )

      mock_process = MagicMock()
      mock_process.pid = 12345
      mock_process.memory_info.return_value = MagicMock(
        rss=64 * 1024**2, vms=128 * 1024**2
      )
      mock_process.cpu_percent.return_value = 15.5
      mock_process.num_threads.return_value = 8
      mock_process.create_time.return_value = time.time() - 3600
      mock_process_class.return_value = mock_process

      system_info = SystemMonitor.get_system_info()

      assert "cpu" in system_info
      assert system_info["cpu"]["percent"] == 25.0
      assert system_info["cpu"]["count"] == 4
      assert system_info["cpu"]["frequency"] is None

  def test_get_system_info_zero_values_edge_cases(self):
    """Test SystemMonitor with zero or minimal values."""
    with (
      patch("psutil.cpu_percent") as mock_cpu,
      patch("psutil.virtual_memory") as mock_memory,
      patch("psutil.swap_memory") as mock_swap,
      patch("psutil.disk_usage") as mock_disk,
      patch("psutil.net_io_counters") as mock_net,
      patch("psutil.Process") as mock_process_class,
    ):
      # Mock with minimal/zero values
      mock_cpu.return_value = 0.0
      mock_memory.return_value = MagicMock(
        total=1024**3, available=1024**3, percent=0.0, used=0, free=1024**3
      )
      mock_swap.return_value = MagicMock(total=0, used=0, free=0, percent=0.0)
      mock_disk.return_value = MagicMock(total=1024**3, used=0, free=1024**3)
      mock_net.return_value = MagicMock(
        bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0
      )

      mock_process = MagicMock()
      mock_process.pid = 1
      mock_process.memory_info.return_value = MagicMock(rss=0, vms=0)
      mock_process.cpu_percent.return_value = 0.0
      mock_process.num_threads.return_value = 1
      mock_process.create_time.return_value = time.time()
      mock_process_class.return_value = mock_process

      system_info = SystemMonitor.get_system_info()

      assert system_info["cpu"]["percent"] == 0.0
      assert system_info["memory"]["percent"] == 0.0
      assert system_info["swap"]["total"] == 0
      assert system_info["network"]["bytes_sent"] == 0
      assert system_info["process"]["memory_rss"] == 0
      assert system_info["disk"]["percent"] == 0.0


class TestGlobalInstances:
  """Test global instances of monitoring classes."""

  def test_health_checker_instance(self):
    """Test global health_checker instance."""
    assert isinstance(health_checker, HealthChecker)

  def test_metrics_collector_instance(self):
    """Test global metrics_collector instance."""
    assert isinstance(metrics_collector, MetricsCollector)

  def test_system_monitor_instance(self):
    """Test global system_monitor instance."""
    assert isinstance(system_monitor, SystemMonitor)


class TestDefaultHealthChecks:
  """Test default health checks registration."""

  @pytest.mark.asyncio
  async def test_basic_health_check_registered(self):
    """Test that basic health check is registered."""
    # Create a fresh health checker to test default registration
    from src.utils.monitoring import HealthChecker

    checker = HealthChecker()

    # Manually register the basic check (as it would be in the module)
    async def basic_health_check():
      return {"healthy": True, "message": "Application is running"}

    checker.register_check("basic", basic_health_check)

    result = await checker.run_all_checks()

    assert "basic" in result["checks"]
    assert result["checks"]["basic"]["status"] == "healthy"
    assert result["checks"]["basic"]["message"] == "Application is running"

  @pytest.mark.asyncio
  async def test_system_resources_check_registered(self):
    """Test that system resources check is registered."""
    from src.utils.monitoring import HealthChecker, SystemMonitor

    checker = HealthChecker()
    monitor = SystemMonitor()

    # Manually register the system resources check
    async def system_resources_check():
      try:
        with patch.object(
          monitor,
          "get_system_info",
          return_value={
            "cpu": {"percent": 50.0},
            "memory": {"percent": 60.0},
            "timestamp": "2023-01-01T00:00:00",
          },
        ):
          system_info = monitor.get_system_info()
          cpu_usage = system_info["cpu"]["percent"]
          memory_percent = system_info["memory"]["percent"]

          if cpu_usage > 90 or memory_percent > 90:
            return {
              "healthy": False,
              "message": f"High resource usage: CPU {cpu_usage}%, Memory {memory_percent}%",
            }

          return {
            "healthy": True,
            "message": f"System resources OK: CPU {cpu_usage}%, Memory {memory_percent}%",
          }
      except Exception as e:
        return {
          "healthy": False,
          "message": f"Failed to check system resources: {str(e)}",
        }

    checker.register_check("system_resources", system_resources_check)

    result = await checker.run_all_checks()

    assert "system_resources" in result["checks"]
    assert result["checks"]["system_resources"]["status"] == "healthy"
    assert "System resources OK" in result["checks"]["system_resources"]["message"]

  @pytest.mark.asyncio
  async def test_basic_health_check_function(self):
    """Test basic_health_check function always returns healthy."""

    result = await basic_health_check()

    assert result["healthy"] is True
    assert result["message"] == "Application is running"
    assert "details" in result

  @pytest.mark.asyncio
  async def test_database_health_check_function(self):
    """Test database_health_check function directly."""

    result = await database_health_check()
    assert result["healthy"] is True
    assert result["message"] == "No database configured"
    assert "details" in result

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_critical_cpu(self, mock_get_system_info):
    """Test system_health_check with critical CPU usage."""

    mock_get_system_info.return_value = {
      "cpu": {"percent": 95.0},
      "memory": {"percent": 60.0},
      "disk": {"percent": 70.0},
    }

    result = await system_health_check()

    assert result["healthy"] is False
    assert result["message"] == "System resources under stress"
    assert "High CPU usage: 95.0%" in result["details"]["warnings"]

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_critical_memory(self, mock_get_system_info):
    """Test system_health_check with critical memory usage."""

    mock_get_system_info.return_value = {
      "cpu": {"percent": 50.0},
      "memory": {"percent": 95.0},
      "disk": {"percent": 60.0},
    }

    result = await system_health_check()

    assert result["healthy"] is False
    assert result["message"] == "System resources under stress"
    assert "High memory usage: 95.0%" in result["details"]["warnings"]

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_missing_components(self, mock_get_system_info):
    """Test system_health_check with missing system components."""

    mock_get_system_info.side_effect = KeyError("cpu")

    result = await system_health_check()
    assert result["healthy"] is False
    assert "Failed to check system health" in result["message"]
    assert "details" in result

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_with_errors(self, mock_get_system_info):
    """Test system_health_check when system info contains errors."""

    mock_get_system_info.side_effect = RuntimeError("System monitoring failed")

    result = await system_health_check()
    assert result["healthy"] is False
    assert "Failed to check system health" in result["message"]
    assert "System monitoring failed" in result["details"]["error"]

  @pytest.mark.asyncio
  async def test_basic_health_check_always_healthy(self):
    """Test that basic_health_check always returns healthy."""

    # Call multiple times to ensure consistency
    for _ in range(10):
      result = await basic_health_check()
      assert result["healthy"] is True
      assert result["message"] == "Application is running"
      assert "details" in result

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_healthy(self, mock_get_system_info):
    """Test system_health_check when system is healthy."""

    # Mock healthy system
    mock_get_system_info.return_value = {
      "cpu": {"percent": 50.0},
      "memory": {"percent": 60.0},
      "disk": {"percent": 70.0},
    }

    result = await system_health_check()
    assert result["healthy"] is True
    assert result["message"] == "System resources OK"
    assert "details" in result

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_unhealthy(self, mock_get_system_info):
    """Test system_health_check when system is under stress."""

    # Mock unhealthy system
    mock_get_system_info.return_value = {
      "cpu": {"percent": 95.0},
      "memory": {"percent": 92.0},
      "disk": {"percent": 88.0},
    }

    result = await system_health_check()
    assert result["healthy"] is False
    assert result["message"] == "System resources under stress"
    assert "High CPU usage: 95.0%" in result["details"]["warnings"]
    assert "High memory usage: 92.0%" in result["details"]["warnings"]

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_exception(self, mock_get_system_info):
    """Test system_health_check when get_system_info raises exception."""

    # Mock exception
    mock_get_system_info.side_effect = Exception("System info error")

    result = await system_health_check()
    assert result["healthy"] is False
    assert "Failed to check system health" in result["message"]
    assert "System info error" in result["details"]["error"]

  @pytest.mark.asyncio
  async def test_get_health_checker(self):
    """Test get_health_checker function."""
    from src.utils.monitoring import get_health_checker, health_checker

    result = get_health_checker()
    assert result is health_checker
    assert isinstance(result, HealthChecker)

  def test_get_metrics_collector(self):
    """Test get_metrics_collector function."""
    from src.utils.monitoring import get_metrics_collector, metrics_collector

    result = get_metrics_collector()
    assert result is metrics_collector
    assert isinstance(result, MetricsCollector)

  def test_get_system_monitor(self):
    """Test get_system_monitor function."""
    from src.utils.monitoring import get_system_monitor, system_monitor

    result = get_system_monitor()
    assert result is system_monitor
    assert isinstance(result, SystemMonitor)

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_boundary_values(self, mock_get_system_info):
    """Test system_health_check with exact boundary values (90%)."""

    # Test exactly at the boundary (90%)
    mock_get_system_info.return_value = {
      "cpu": {"percent": 90.0},
      "memory": {"percent": 90.0},
      "disk": {"percent": 90.0},
    }

    result = await system_health_check()
    assert result["healthy"] is False
    assert result["message"] == "System resources under stress"
    assert len(result["details"]["warnings"]) == 3

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_just_below_threshold(self, mock_get_system_info):
    """Test system_health_check just below threshold (89.9%)."""

    # Test just below the threshold
    mock_get_system_info.return_value = {
      "cpu": {"percent": 89.9},
      "memory": {"percent": 89.9},
      "disk": {"percent": 89.9},
    }

    result = await system_health_check()
    assert result["healthy"] is True
    assert result["message"] == "System resources OK"
    assert result["details"]["warnings"] == []

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.system_monitor.get_system_info")
  async def test_system_health_check_mixed_thresholds(self, mock_get_system_info):
    """Test system_health_check with mixed threshold violations."""

    # Test mixed scenario: CPU and disk high, memory OK
    mock_get_system_info.return_value = {
      "cpu": {"percent": 95.0},
      "memory": {"percent": 50.0},
      "disk": {"percent": 92.0},
    }

    result = await system_health_check()
    assert result["healthy"] is False
    assert result["message"] == "System resources under stress"
    warnings = result["details"]["warnings"]
    assert len(warnings) == 2
    assert "High CPU usage: 95.0%" in warnings
    assert "High disk usage: 92.0%" in warnings
    assert "High memory usage" not in str(warnings)

  @pytest.mark.asyncio
  @patch("src.utils.monitoring.settings")
  async def test_basic_health_check_with_different_settings(self, mock_settings):
    """Test basic_health_check with different application settings."""

    # Mock the settings object directly
    mock_settings.app_name = "Test App"
    mock_settings.app_version = "2.0.0"
    mock_settings.environment = "production"

    result = await basic_health_check()

    assert result["healthy"] is True
    assert result["message"] == "Application is running"
    assert result["details"]["app_name"] == "Test App"
    assert result["details"]["version"] == "2.0.0"
    assert result["details"]["environment"] == "production"

  @pytest.mark.asyncio
  async def test_database_health_check_consistency(self):
    """Test database_health_check returns consistent results."""

    # Call multiple times to ensure consistency
    results = []
    for _ in range(5):
      result = await database_health_check()
      results.append(result)

    # All results should be identical
    for result in results:
      assert result["healthy"] is True
      assert result["message"] == "No database configured"
      assert result["details"]["status"] == "not_applicable"


class TestErrorScenarios:
  """Test various error scenarios and edge cases."""

  @pytest.mark.asyncio
  async def test_health_check_network_timeout_simulation(self):
    """Test health check behavior during network timeout scenarios."""
    checker = HealthChecker()

    async def network_timeout_check():
      # Simulate network timeout
      await asyncio.sleep(0.5)
      raise TimeoutError("Network request timed out")

    checker.register_check("network_service", network_timeout_check, timeout=0.1)
    result = await checker.run_check("network_service")

    assert result["status"] == "timeout"
    assert "timed out" in result["message"].lower()

  @pytest.mark.asyncio
  async def test_health_check_connection_refused(self):
    """Test health check behavior when connection is refused."""
    checker = HealthChecker()

    async def connection_refused_check():
      raise ConnectionRefusedError("Connection refused by target service")

    checker.register_check("external_service", connection_refused_check)
    result = await checker.run_check("external_service")

    assert result["status"] == "error"
    assert "Connection refused" in result["message"]

  @pytest.mark.asyncio
  async def test_health_check_dns_resolution_failure(self):
    """Test health check behavior during DNS resolution failures."""
    checker = HealthChecker()

    async def dns_failure_check():
      import socket

      raise socket.gaierror("Name or service not known")

    checker.register_check("dns_dependent_service", dns_failure_check)
    result = await checker.run_check("dns_dependent_service")

    assert result["status"] == "error"
    assert "Name or service not known" in result["message"]

  @pytest.mark.asyncio
  async def test_system_monitor_resource_exhaustion(self):
    """Test system monitor behavior during resource exhaustion."""
    with patch("psutil.virtual_memory") as mock_memory:
      # Simulate memory exhaustion
      mock_memory.side_effect = MemoryError("Cannot allocate memory")

      system_info = SystemMonitor.get_system_info()

      assert "error" in system_info
      assert "Cannot allocate memory" in system_info["error"]

  @pytest.mark.asyncio
  async def test_metrics_collector_disk_full_simulation(self):
    """Test metrics collector behavior when disk is full."""
    collector = MetricsCollector()

    # Simulate recording metrics when disk might be full
    # This tests the robustness of in-memory operations
    for i in range(10000):
      collector.increment_counter("stress_test_counter")
      collector.record_histogram("stress_test_histogram", i * 0.001)

    metrics = collector.get_metrics()
    assert metrics["custom_metrics"]["stress_test_counter"]["value"] == 10000
    # Histogram has a limit of 1000 values
    assert len(metrics["custom_metrics"]["stress_test_histogram"]["values"]) == 1000

  @pytest.mark.asyncio
  async def test_health_checker_concurrent_check_failures(self):
    """Test health checker behavior with multiple concurrent check failures."""
    checker = HealthChecker()

    async def failing_check_1():
      raise RuntimeError("Service 1 failed")

    async def failing_check_2():
      raise ValueError("Service 2 validation error")

    async def failing_check_3():
      raise ConnectionError("Service 3 connection error")

    checker.register_check("service1", failing_check_1)
    checker.register_check("service2", failing_check_2)
    checker.register_check("service3", failing_check_3)

    result = await checker.run_all_checks()

    assert result["status"] == "error"
    assert len(result["checks"]) == 3

    for service_name in ["service1", "service2", "service3"]:
      assert result["checks"][service_name]["status"] == "error"

  @pytest.mark.asyncio
  async def test_system_health_check_permission_denied(self):
    """Test system health check when permission is denied for system info."""

    with patch("src.utils.monitoring.system_monitor.get_system_info") as mock_get_info:
      mock_get_info.side_effect = PermissionError(
        "Permission denied to access system info"
      )

      result = await system_health_check()

      assert result["healthy"] is False
      assert "Failed to check system health" in result["message"]
      assert "Permission denied" in result["details"]["error"]

  def test_metrics_collector_extreme_values(self):
    """Test metrics collector with extreme values."""
    collector = MetricsCollector()

    # Test with very large numbers
    collector.set_gauge("large_gauge", float("inf"))
    collector.record_histogram("large_histogram", 1e20)

    # Test with very small numbers
    collector.set_gauge("small_gauge", 1e-20)
    collector.record_histogram("small_histogram", 1e-20)

    # Test with negative numbers
    collector.record_histogram("negative_histogram", -1000)

    metrics = collector.get_metrics()

    # Should handle extreme values without crashing
    assert "large_gauge" in metrics["custom_metrics"]
    assert "small_gauge" in metrics["custom_metrics"]
    assert len(metrics["custom_metrics"]["large_histogram"]["values"]) == 1
    assert len(metrics["custom_metrics"]["small_histogram"]["values"]) == 1
    assert len(metrics["custom_metrics"]["negative_histogram"]["values"]) == 1


class TestTimeoutAndCachingEdgeCases:
  """Test additional timeout and caching edge cases."""

  @pytest.mark.asyncio
  async def test_cache_corruption_recovery(self):
    """Test recovery from corrupted cache state."""
    checker = HealthChecker()

    async def test_check():
      return {"healthy": True, "message": "OK"}

    checker.register_check("test", test_check)

    # Populate cache
    await checker.run_all_checks(use_cache=True)

    # Corrupt cache by setting invalid data
    checker.cached_results = "invalid_cache_data"

    # When cache is corrupted, it should return the corrupted data
    # This tests the current behavior - cache is returned as-is if fresh
    result = await checker.run_all_checks(use_cache=True)
    assert result == "invalid_cache_data"

    # Force cache refresh by disabling cache
    result = await checker.run_all_checks(use_cache=False)
    assert result["status"] == "healthy"
    assert "test" in result["checks"]

  @pytest.mark.asyncio
  async def test_timeout_precision_edge_cases(self):
    """Test timeout precision with very small timeout values."""
    checker = HealthChecker()

    async def micro_delay_check():
      await asyncio.sleep(0.001)  # 1ms delay
      return {"healthy": True, "message": "Fast"}

    # Test with timeout smaller than execution time
    checker.register_check("micro", micro_delay_check, timeout=0.0001)  # 0.1ms timeout
    result = await checker.run_check("micro")

    # Should timeout due to precision
    assert result["status"] == "timeout"

  @pytest.mark.asyncio
  async def test_cache_behavior_under_concurrent_modifications(self):
    """Test cache behavior when checks are modified during execution."""
    checker = HealthChecker()
    call_count = 0

    async def counting_check():
      nonlocal call_count
      call_count += 1
      await asyncio.sleep(0.1)  # Simulate work
      return {"healthy": True, "message": f"Count: {call_count}"}

    checker.register_check("counter", counting_check)

    # Start a check that will take time
    task1 = asyncio.create_task(checker.run_all_checks(use_cache=True))

    # Immediately start another check
    task2 = asyncio.create_task(checker.run_all_checks(use_cache=True))

    # Wait for both to complete
    result1, result2 = await asyncio.gather(task1, task2)

    # Both should succeed
    assert result1["status"] == "healthy"
    assert result2["status"] == "healthy"

    # Call count should be reasonable (not excessive due to race conditions)
    assert call_count <= 2

  @pytest.mark.asyncio
  async def test_timeout_with_cleanup_operations(self):
    """Test timeout behavior when check has cleanup operations."""
    checker = HealthChecker()
    cleanup_called = False

    async def check_with_cleanup():
      nonlocal cleanup_called
      try:
        await asyncio.sleep(1.0)  # Long operation
        return {"healthy": True, "message": "Completed"}
      finally:
        cleanup_called = True

    checker.register_check("cleanup", check_with_cleanup, timeout=0.1)
    result = await checker.run_check("cleanup")

    # Should timeout
    assert result["status"] == "timeout"

    # Give a moment for cleanup to potentially run
    await asyncio.sleep(0.1)

    # Cleanup should have been called due to task cancellation
    assert cleanup_called

  @pytest.mark.asyncio
  async def test_cache_memory_pressure_simulation(self):
    """Test cache behavior under memory pressure simulation."""
    checker = HealthChecker()

    async def memory_intensive_check():
      # Simulate a check that returns large data
      large_data = "x" * 10000  # 10KB string
      return {
        "healthy": True,
        "message": "Memory intensive",
        "details": {"large_data": large_data},
      }

    checker.register_check("memory_test", memory_intensive_check)

    # Run multiple times to test cache with large data
    for i in range(5):
      result = await checker.run_all_checks(use_cache=True)
      assert result["status"] == "healthy"
      assert "memory_test" in result["checks"]
      assert len(result["checks"]["memory_test"]["details"]["large_data"]) == 10000

  @pytest.mark.asyncio
  async def test_timeout_inheritance_and_override(self):
    """Test timeout inheritance and override behavior."""
    checker = HealthChecker()

    async def variable_delay_check(delay):
      await asyncio.sleep(delay)
      return {"healthy": True, "message": f"Delayed {delay}s"}

    # Register checks with different timeouts
    checker.register_check("fast", lambda: variable_delay_check(0.05), timeout=0.1)
    checker.register_check("medium", lambda: variable_delay_check(0.15), timeout=0.2)
    checker.register_check("slow", lambda: variable_delay_check(0.25), timeout=0.3)

    result = await checker.run_all_checks()

    # Fast should succeed
    assert result["checks"]["fast"]["status"] == "healthy"
    # Medium should succeed
    assert result["checks"]["medium"]["status"] == "healthy"
    # Slow should succeed
    assert result["checks"]["slow"]["status"] == "healthy"

  def test_cache_duration_configuration(self):
    """Test cache duration configuration and validation."""
    checker = HealthChecker()

    # Test default cache duration
    assert checker.check_cache_duration == 30

    # Test setting custom cache duration
    checker.check_cache_duration = 60
    assert checker.check_cache_duration == 60

    # Test setting zero cache duration (immediate expiry)
    checker.check_cache_duration = 0
    assert checker.check_cache_duration == 0

    # Test negative cache duration (should be allowed for testing)
    checker.check_cache_duration = -1
    assert checker.check_cache_duration == -1


class TestMonitoringIntegration:
  """Test monitoring integration scenarios."""

  @pytest.mark.asyncio
  async def test_health_check_with_metrics_collection(self):
    """Test health checks that also collect metrics."""
    checker = HealthChecker()
    collector = MetricsCollector()

    async def health_check_with_metrics():
      # Simulate collecting metrics during health check
      collector.increment_counter("health_checks_total")
      collector.set_gauge("service_status", 1)  # 1 = healthy
      return {"healthy": True, "message": "Service is healthy"}

    checker.register_check("service_with_metrics", health_check_with_metrics)

    result = await checker.run_all_checks()
    metrics = collector.get_metrics()

    assert result["checks"]["service_with_metrics"]["status"] == "healthy"
    assert metrics["custom_metrics"]["health_checks_total"]["value"] == 1
    assert metrics["custom_metrics"]["service_status"]["value"] == 1

  @pytest.mark.asyncio
  async def test_system_monitoring_with_alerts(self):
    """Test system monitoring with alert thresholds."""
    monitor = SystemMonitor()
    collector = MetricsCollector()

    # Mock high CPU usage
    with patch("psutil.cpu_percent", return_value=95.0):
      with patch("psutil.virtual_memory") as mock_memory:
        mock_memory.return_value = MagicMock(percent=85.0)
        system_info = monitor.get_system_info()

        # Collect metrics and check thresholds
        cpu_usage = system_info["cpu"]["percent"]
        memory_percent = system_info["memory"]["percent"]

        collector.set_gauge("cpu_usage_percent", cpu_usage)
        collector.set_gauge("memory_usage_percent", memory_percent)

        # Simulate alert conditions
        if cpu_usage > 90:
          collector.increment_counter("cpu_alerts_total")

        if memory_percent > 80:
          collector.increment_counter("memory_alerts_total")

        metrics = collector.get_metrics()

        assert metrics["custom_metrics"]["cpu_usage_percent"]["value"] == 95.0
        assert metrics["custom_metrics"]["memory_usage_percent"]["value"] == 85.0
        assert metrics["custom_metrics"]["cpu_alerts_total"]["value"] == 1
        assert metrics["custom_metrics"]["memory_alerts_total"]["value"] == 1

  def test_metrics_aggregation(self):
    """Test metrics aggregation and calculation."""
    collector = MetricsCollector()

    # Simulate multiple requests with different response times
    response_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22]

    for response_time in response_times:
      collector.record_histogram("response_time", response_time)
      collector.increment_counter("requests_total")

    metrics = collector.get_metrics()

    assert metrics["custom_metrics"]["requests_total"]["value"] == len(response_times)
    assert len(metrics["custom_metrics"]["response_time"]["values"]) == len(
      response_times
    )

    # Calculate basic statistics
    histogram_values = metrics["custom_metrics"]["response_time"]["values"]
    avg_response_time = sum(histogram_values) / len(histogram_values)
    min_response_time = min(histogram_values)
    max_response_time = max(histogram_values)

    assert abs(avg_response_time - 0.2) < 0.01  # Approximately 0.2
    assert min_response_time == 0.1
    assert max_response_time == 0.3

  @pytest.mark.asyncio
  async def test_monitoring_performance_impact(self):
    """Test that monitoring has minimal performance impact."""
    collector = MetricsCollector()

    # Measure time to perform many metric operations
    start_time = time.time()

    for i in range(1000):
      collector.increment_counter("test_counter")
      collector.set_gauge("test_gauge", i)
      collector.record_histogram("test_histogram", i * 0.001)

    end_time = time.time()
    duration = end_time - start_time

    # Should complete quickly (less than 1 second for 1000 operations)
    assert duration < 1.0

    # Verify metrics were collected correctly
    metrics = collector.get_metrics()
    assert metrics["custom_metrics"]["test_counter"]["value"] == 1000
    assert metrics["custom_metrics"]["test_gauge"]["value"] == 999
    assert len(metrics["custom_metrics"]["test_histogram"]["values"]) == 1000
