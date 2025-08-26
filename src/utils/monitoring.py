"""Monitoring and health check utilities for the editor agent application."""

import asyncio
import psutil
import time
from datetime import datetime, UTC
from typing import Any, Dict

from src.config.settings import get_settings
from src.utils.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HealthChecker:
  """Health check manager for various system components."""

  def __init__(self):
    self.checks = {}
    self.last_check_time = None
    self.check_cache_duration = 30  # seconds
    self.cached_results = None

  def register_check(self, name: str, check_func, timeout: float = 5.0):
    """Register a health check function."""
    self.checks[name] = {"func": check_func, "timeout": timeout}
    logger.info(f"Registered health check: {name}")

  async def run_check(self, name: str) -> Dict[str, Any]:
    """Run a single health check."""
    if name not in self.checks:
      return {
        "status": "error",
        "message": f"Health check '{name}' not found",
        "timestamp": datetime.now(UTC).isoformat(),
      }

    check_info = self.checks[name]
    start_time = time.time()

    try:
      # Run check with timeout
      result = await asyncio.wait_for(
        check_info["func"](), timeout=check_info["timeout"]
      )

      duration = time.time() - start_time

      return {
        "status": "healthy" if result.get("healthy", True) else "unhealthy",
        "message": result.get("message", "OK"),
        "details": result.get("details", {}),
        "duration": duration,
        "timestamp": datetime.now(UTC).isoformat(),
      }

    except asyncio.TimeoutError:
      duration = time.time() - start_time
      return {
        "status": "timeout",
        "message": f"Health check timed out after {check_info['timeout']}s",
        "duration": duration,
        "timestamp": datetime.now(UTC).isoformat(),
      }

    except Exception as e:
      duration = time.time() - start_time
      logger.error(f"Health check '{name}' failed: {str(e)}", exc_info=True)
      return {
        "status": "error",
        "message": f"Health check failed: {str(e)}",
        "duration": duration,
        "timestamp": datetime.now(UTC).isoformat(),
      }

  async def run_all_checks(self, use_cache: bool = True) -> Dict[str, Any]:
    """Run all registered health checks."""
    current_time = time.time()

    # Return cached results if available and fresh
    if (
      use_cache
      and self.cached_results
      and self.last_check_time
      and current_time - self.last_check_time < self.check_cache_duration
    ):
      return self.cached_results

    start_time = time.time()
    results = {}

    # Run all checks concurrently
    tasks = {name: self.run_check(name) for name in self.checks.keys()}

    if tasks:
      completed_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
      for name, result in zip(tasks.keys(), completed_results):
        if isinstance(result, Exception):
          results[name] = {
            "status": "error",
            "message": f"Check execution failed: {str(result)}",
            "timestamp": datetime.now(UTC).isoformat(),
          }
        else:
          results[name] = result

    # Calculate overall status
    overall_status = "healthy"
    unhealthy_count = 0
    error_count = 0

    for check_result in results.values():
      if check_result["status"] == "unhealthy":
        unhealthy_count += 1
      elif check_result["status"] in ["error", "timeout"]:
        error_count += 1

    if error_count > 0:
      overall_status = "error"
    elif unhealthy_count > 0:
      overall_status = "unhealthy"

    total_duration = time.time() - start_time

    health_report = {
      "status": overall_status,
      "timestamp": datetime.now(UTC).isoformat(),
      "duration": total_duration,
      "checks": results,
      "summary": {
        "total_checks": len(results),
        "healthy_checks": len(
          [r for r in results.values() if r["status"] == "healthy"]
        ),
        "unhealthy_checks": unhealthy_count,
        "error_checks": error_count,
      },
    }

    # Cache results
    self.cached_results = health_report
    self.last_check_time = current_time

    return health_report


class MetricsCollector:
  """Collect and manage application metrics."""

  def __init__(self):
    self.metrics = {}
    self.start_time = time.time()
    self.request_count = 0
    self.error_count = 0
    self.response_times = []
    self.max_response_times = 1000  # Keep last 1000 response times

  def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
    """Increment a counter metric."""
    key = self._get_metric_key(name, labels)
    if key not in self.metrics:
      self.metrics[key] = {"type": "counter", "value": 0, "labels": labels or {}}
    self.metrics[key]["value"] += value

  def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
    """Set a gauge metric value."""
    key = self._get_metric_key(name, labels)
    self.metrics[key] = {"type": "gauge", "value": value, "labels": labels or {}}

  def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
    """Record a histogram value."""
    key = self._get_metric_key(name, labels)
    if key not in self.metrics:
      self.metrics[key] = {"type": "histogram", "values": [], "labels": labels or {}}

    self.metrics[key]["values"].append(value)
    # Keep only recent values to prevent memory growth
    if len(self.metrics[key]["values"]) > 1000:
      self.metrics[key]["values"] = self.metrics[key]["values"][-1000:]

  def _get_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
    """Generate a unique key for the metric."""
    if not labels:
      return name
    label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"

  def record_request(
    self, method: str, endpoint: str, status_code: int, duration: float
  ):
    """Record HTTP request metrics."""
    self.request_count += 1

    # Record request count
    self.increment_counter(
      "http_requests_total",
      labels={"method": method, "endpoint": endpoint, "status": str(status_code)},
    )

    # Record response time
    self.record_histogram(
      "http_request_duration_seconds",
      duration,
      labels={"method": method, "endpoint": endpoint},
    )

    # Track response times for quick stats
    self.response_times.append(duration)
    if len(self.response_times) > self.max_response_times:
      self.response_times = self.response_times[-self.max_response_times :]

    # Count errors
    if status_code >= 400:
      self.error_count += 1
      self.increment_counter(
        "http_errors_total",
        labels={"method": method, "endpoint": endpoint, "status": str(status_code)},
      )

  def get_metrics(self) -> Dict[str, Any]:
    """Get all collected metrics."""
    uptime = time.time() - self.start_time

    # Calculate response time statistics
    response_time_stats = {}
    if self.response_times:
      sorted_times = sorted(self.response_times)
      response_time_stats = {
        "min": min(sorted_times),
        "max": max(sorted_times),
        "avg": sum(sorted_times) / len(sorted_times),
        "p50": sorted_times[int(len(sorted_times) * 0.5)],
        "p95": sorted_times[int(len(sorted_times) * 0.95)],
        "p99": sorted_times[int(len(sorted_times) * 0.99)],
      }

    return {
      "uptime_seconds": uptime,
      "total_requests": self.request_count,
      "total_errors": self.error_count,
      "error_rate": self.error_count / max(self.request_count, 1),
      "response_time_stats": response_time_stats,
      "custom_metrics": self.metrics,
      "timestamp": datetime.now(UTC).isoformat(),
    }


class SystemMonitor:
  """Monitor system resources and performance."""

  @staticmethod
  def get_system_info() -> Dict[str, Any]:
    """Get current system information."""
    try:
      # CPU information
      cpu_percent = psutil.cpu_percent(interval=1)
      cpu_count = psutil.cpu_count()
      cpu_freq = psutil.cpu_freq()

      # Memory information
      memory = psutil.virtual_memory()
      swap = psutil.swap_memory()

      # Disk information
      disk = psutil.disk_usage("/")

      # Network information
      network = psutil.net_io_counters()

      # Process information
      process = psutil.Process()
      process_memory = process.memory_info()

      return {
        "cpu": {
          "percent": cpu_percent,
          "count": cpu_count,
          "frequency": {
            "current": cpu_freq.current if cpu_freq else None,
            "min": cpu_freq.min if cpu_freq else None,
            "max": cpu_freq.max if cpu_freq else None,
          }
          if cpu_freq
          else None,
        },
        "memory": {
          "total": memory.total,
          "available": memory.available,
          "percent": memory.percent,
          "used": memory.used,
          "free": memory.free,
        },
        "swap": {
          "total": swap.total,
          "used": swap.used,
          "free": swap.free,
          "percent": swap.percent,
        },
        "disk": {
          "total": disk.total,
          "used": disk.used,
          "free": disk.free,
          "percent": (disk.used / disk.total) * 100,
        },
        "network": {
          "bytes_sent": network.bytes_sent,
          "bytes_recv": network.bytes_recv,
          "packets_sent": network.packets_sent,
          "packets_recv": network.packets_recv,
        },
        "process": {
          "pid": process.pid,
          "memory_rss": process_memory.rss,
          "memory_vms": process_memory.vms,
          "cpu_percent": process.cpu_percent(),
          "num_threads": process.num_threads(),
          "create_time": process.create_time(),
        },
        "timestamp": datetime.now(UTC).isoformat(),
      }

    except Exception as e:
      logger.error(f"Failed to get system info: {str(e)}", exc_info=True)
      return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}


# Global instances
health_checker = HealthChecker()
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor()


# Default health checks
async def basic_health_check() -> Dict[str, Any]:
  """Basic application health check."""
  return {
    "healthy": True,
    "message": "Application is running",
    "details": {
      "app_name": settings.app_name,
      "version": settings.app_version,
      "environment": settings.environment,
    },
  }


async def system_health_check() -> Dict[str, Any]:
  """System resource health check."""
  try:
    system_info = system_monitor.get_system_info()

    # Check if system resources are healthy
    cpu_healthy = system_info["cpu"]["percent"] < 90
    memory_healthy = system_info["memory"]["percent"] < 90
    disk_healthy = system_info["disk"]["percent"] < 90

    overall_healthy = cpu_healthy and memory_healthy and disk_healthy

    warnings = []
    if not cpu_healthy:
      warnings.append(f"High CPU usage: {system_info['cpu']['percent']:.1f}%")
    if not memory_healthy:
      warnings.append(f"High memory usage: {system_info['memory']['percent']:.1f}%")
    if not disk_healthy:
      warnings.append(f"High disk usage: {system_info['disk']['percent']:.1f}%")

    return {
      "healthy": overall_healthy,
      "message": "System resources OK"
      if overall_healthy
      else "System resources under stress",
      "details": {
        "warnings": warnings,
        "cpu_percent": system_info["cpu"]["percent"],
        "memory_percent": system_info["memory"]["percent"],
        "disk_percent": system_info["disk"]["percent"],
      },
    }

  except Exception as e:
    return {
      "healthy": False,
      "message": f"Failed to check system health: {str(e)}",
      "details": {"error": str(e)},
    }


async def database_health_check() -> Dict[str, Any]:
  """Database health check (placeholder for future database integration)."""
  # This is a placeholder - implement actual database health check when database is added
  return {
    "healthy": True,
    "message": "No database configured",
    "details": {"status": "not_applicable"},
  }


# Register default health checks
health_checker.register_check("basic", basic_health_check, timeout=2.0)
health_checker.register_check("system", system_health_check, timeout=5.0)
health_checker.register_check("database", database_health_check, timeout=10.0)


def get_health_checker() -> HealthChecker:
  """Get the global health checker instance."""
  return health_checker


def get_metrics_collector() -> MetricsCollector:
  """Get the global metrics collector instance."""
  return metrics_collector


def get_system_monitor() -> SystemMonitor:
  """Get the global system monitor instance."""
  return system_monitor
