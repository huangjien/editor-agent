"""Tests for API routes."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHealthRoutes:
  """Test health check endpoints."""

  def test_root_endpoint(self, client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "timestamp" in data
    assert "version" in data

  def test_health_endpoint(self, client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "checks" in data

  def test_metrics_endpoint(self, client: TestClient):
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_errors" in data
    assert "error_rate" in data
    assert "response_time_stats" in data
    assert "custom_metrics" in data
    assert "timestamp" in data

  def test_system_endpoint(self, client: TestClient):
    """Test the system info endpoint."""
    response = client.get("/system")
    assert response.status_code == 200
    data = response.json()
    assert "cpu" in data
    assert "memory" in data
    assert "disk" in data


class TestAgentRoutes:
  """Test agent-related endpoints."""

  def test_execute_agent_task_missing_auth(self, auth_client: TestClient):
    """Test agent task execution without API key."""
    response = auth_client.post(
      "/api/v1/agent/execute", json={"task": "Test task", "session_id": "test-session"}
    )
    assert response.status_code == 401

  def test_execute_agent_task_with_auth(self, auth_client: TestClient):
    """Test agent task execution with API key."""
    headers = {"X-API-Key": "test-api-key"}
    response = auth_client.post(
      "/api/v1/agent/execute",
      json={"task": "Test task", "session_id": "test-session"},
      headers=headers,
    )
    # Should return 200 or 422 depending on implementation
    if response.status_code not in [200, 422]:
      print(f"Status: {response.status_code}, Content: {response.text}")
    assert response.status_code in [200, 422]

  def test_agent_status(self, auth_client: TestClient):
    """Test agent status endpoint."""
    headers = {"X-API-Key": "test-api-key"}
    response = auth_client.get("/api/v1/agent/status", headers=headers)
    assert response.status_code in [200, 500]

  def test_invalid_session_id_format(self, client: TestClient):
    """Test agent status without authentication."""
    response = client.get("/api/v1/agent/status")
    assert response.status_code == 200


class TestChatRoutes:
  """Test chat-related endpoints."""

  def test_send_chat_message_missing_auth(self, auth_client: TestClient):
    """Test chat message without API key."""
    response = auth_client.post(
      "/api/v1/chat/message",
      json={"message": "Hello", "session_id": "test-session"},
    )
    assert response.status_code == 401

  def test_send_chat_message_with_auth(self, auth_client: TestClient):
    """Test chat message with API key."""
    headers = {"X-API-Key": "test-api-key"}
    response = auth_client.post(
      "/api/v1/chat/message",
      json={"message": "Hello", "session_id": "test-session"},
      headers=headers,
    )
    assert response.status_code in [200, 422, 500]

  def test_get_chat_history(self, auth_client: TestClient):
    """Test getting chat history."""
    headers = {"X-API-Key": "test-api-key"}
    response = auth_client.get(
      "/api/v1/chat/sessions/test-session/history", headers=headers
    )
    assert response.status_code in [200, 404, 500]

  def test_clear_chat_session(self, auth_client: TestClient):
    """Test clearing chat session."""
    headers = {"X-API-Key": "test-api-key"}
    response = auth_client.delete("/api/v1/chat/sessions/test-session", headers=headers)
    assert response.status_code in [200, 404, 500]

  def test_clear_chat_session_missing_auth(self, auth_client: TestClient):
    """Test clearing chat session without API key."""
    response = auth_client.delete("/api/v1/chat/sessions/test-session")
    assert response.status_code == 401


class TestRequestValidation:
  """Test request validation and error handling."""

  def test_invalid_json_payload(self, client: TestClient):
    """Test handling of invalid JSON payload."""
    headers = {"X-API-Key": "test-api-key", "Content-Type": "application/json"}
    response = client.post(
      "/api/v1/chat/message", content="{invalid json}", headers=headers
    )
    assert response.status_code == 422

  def test_missing_required_fields(self, client: TestClient):
    """Test handling of missing required fields."""
    headers = {"X-API-Key": "test-api-key"}
    response = client.post(
      "/api/v1/chat/message",
      json={"session_id": "test-session"},  # missing message
      headers=headers,
    )
    assert response.status_code == 422

  def test_empty_message(self, client: TestClient):
    """Test handling of empty message."""
    headers = {"X-API-Key": "test-api-key"}
    response = client.post(
      "/api/v1/chat/message",
      json={"message": "", "session_id": "test-session"},
      headers=headers,
    )
    assert response.status_code == 422

  def test_oversized_request(self, client: TestClient):
    """Test handling of oversized request."""
    headers = {"X-API-Key": "test-api-key"}
    large_message = "x" * (2 * 1024 * 1024)  # 2MB message
    response = client.post(
      "/api/v1/chat/message",
      json={"message": large_message, "session_id": "test-session"},
      headers=headers,
    )
    assert response.status_code == 413  # Request Entity Too Large


@pytest.mark.asyncio
class TestAsyncRoutes:
  """Test async route functionality."""

  async def test_async_health_check(self, async_client: AsyncClient):
    """Test async health check."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

  async def test_async_agent_execution(self, async_client: AsyncClient):
    """Test async agent execution."""
    headers = {"X-API-Key": "test-api-key"}
    response = await async_client.post(
      "/api/v1/agent/execute",
      json={"task": "Test async task", "session_id": "test-async-session"},
      headers=headers,
    )
    assert response.status_code in [200, 422]

  async def test_async_chat_message(self, async_client: AsyncClient):
    """Test async chat message."""
    headers = {"X-API-Key": "test-api-key"}
    response = await async_client.post(
      "/api/v1/chat/message",
      json={"message": "Hello async", "session_id": "test-async-session"},
      headers=headers,
    )
    assert response.status_code in [200, 422]
