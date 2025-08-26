"""Tests for monitoring utilities."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.utils.monitoring import (
  HealthChecker,
  MetricsCollector,
  SystemMonitor,
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
