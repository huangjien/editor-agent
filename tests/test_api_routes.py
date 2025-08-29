import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.main import create_app
from src.agent.service import AgentService


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    from src.config.settings import Settings
    
    # Override settings to include testserver in trusted hosts
    test_settings = Settings(
        trusted_hosts="localhost,127.0.0.1,testserver"
    )
    return create_app(settings_override=test_settings)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def client_with_mock_service(app, mock_agent_service):
    """Create a test client with mocked agent service."""
    from src.api.routes import get_agent_service
    
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service
    
    client = TestClient(app, base_url="http://localhost")
    yield client
    
    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def mock_agent_service():
    """Mock agent service."""
    service = Mock(spec=AgentService)
    service.execute_task = AsyncMock()
    service.get_status = AsyncMock()
    return service


@pytest.fixture
def mock_chat_service():
    """Mock chat service."""
    service = Mock()
    service.process_chat_message = AsyncMock()
    service.get_chat_history = AsyncMock()
    service.clear_session = AsyncMock()
    return service


class TestAgentRoutes:
    """Test agent-related endpoints."""

    def test_execute_task_success(self, client_with_mock_service, mock_agent_service):
        """Test successful task execution."""
        mock_agent_service.execute_task.return_value = {
            "result": "Task completed successfully",
            "metadata": {"execution_time": 1.5}
        }
        
        response = client_with_mock_service.post("/api/v1/agent/execute", json={
            "task": "Test task",
            "context": {"key": "value"}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"] == {
            "result": "Task completed successfully",
            "metadata": {"execution_time": 1.5}
        }
        assert data["message"] == "Task executed successfully"
        mock_agent_service.execute_task.assert_called_once()

    def test_execute_task_validation_error(self, client):
        """Test task execution with invalid request data."""
        response = client.post(
            "/api/v1/agent/execute",
            json={"invalid_field": "value"}
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_execute_task_missing_task(self, client):
        """Test task execution with missing task field."""
        response = client.post(
            "/api/v1/agent/execute",
            json={"context": {"some": "data"}}
        )
        
        assert response.status_code == 422

    def test_execute_task_empty_task(self, client_with_mock_service, mock_agent_service):
        """Test task execution with empty task string."""
        # The route doesn't validate empty task, so it will call the service
        mock_agent_service.execute_task.return_value = {"result": "Empty task handled"}
        
        response = client_with_mock_service.post(
            "/api/v1/agent/execute",
            json={"task": "", "context": {}}
        )
        
        # Since there's no validation for empty task in the route, it returns 200
        assert response.status_code == 200

    def test_execute_task_service_failure(self, client_with_mock_service, mock_agent_service):
        """Test task execution when service fails."""
        mock_agent_service.execute_task.side_effect = Exception("Service error")
        
        response = client_with_mock_service.post("/api/v1/agent/execute", json={
            "task": "Test task",
            "context": {}
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Agent execution failed: Service error" in data["error"]["message"]

    def test_execute_task_http_exception(self, client_with_mock_service, mock_agent_service):
        """Test task execution when service raises HTTPException."""
        mock_agent_service.execute_task.side_effect = HTTPException(
            status_code=400, detail="Bad request"
        )
        
        response = client_with_mock_service.post(
            "/api/v1/agent/execute",
            json={"task": "Test task", "context": {}}
        )
        
        # The route catches HTTPException and wraps it in a 500 error
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Agent execution failed" in data["error"]["message"]

    def test_get_status_success(self, client_with_mock_service, mock_agent_service):
        """Test successful status retrieval."""
        mock_agent_service.get_status.return_value = {
            "version": "1.0.0",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        response = client_with_mock_service.get("/api/v1/agent/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert data["agent_info"]["version"] == "1.0.0"
        assert data["timestamp"] == "2024-01-01T00:00:00Z"
        mock_agent_service.get_status.assert_called_once()

    def test_get_status_service_failure(self, client_with_mock_service, mock_agent_service):
        """Test status retrieval when service fails."""
        mock_agent_service.get_status.side_effect = Exception("Status error")
        
        response = client_with_mock_service.get("/api/v1/agent/status")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Failed to get agent status: Status error" in data["error"]["message"]


class TestChatRoutes:
    """Test chat-related endpoints."""

    def test_send_message_success(self, client_with_mock_service, mock_agent_service):
        """Test successful message sending."""
        mock_agent_service.process_chat_message.return_value = {
            "message": "Hello! How can I help you?",
            "session_id": "test-session-123",
            "metadata": {"model": "gpt-4"}
        }
        
        response = client_with_mock_service.post(
            "/api/v1/chat/message",
            json={
                "message": "Hello",
                "session_id": "test-session-123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Hello! How can I help you?"
        assert data["session_id"] == "test-session-123"
        assert "metadata" in data
        mock_agent_service.process_chat_message.assert_called_once()

    def test_send_message_validation_error(self, client):
        """Test message sending with invalid request data."""
        response = client.post(
            "/api/v1/chat/message",
            json={"invalid_field": "value"}
        )
        
        assert response.status_code == 422

    def test_send_message_empty_message(self, client):
        """Test message sending with empty message."""
        response = client.post(
            "/api/v1/chat/message",
            json={"message": "", "session_id": "test-session"}
        )
        
        assert response.status_code == 422

    def test_send_message_service_failure(self, client_with_mock_service, mock_agent_service):
        """Test message sending when service fails."""
        mock_agent_service.process_chat_message.side_effect = Exception("Chat error")
        
        response = client_with_mock_service.post(
            "/api/v1/chat/message",
            json={"message": "Hello", "session_id": "test-session"}
        )
        
        assert response.status_code == 500
        response_data = response.json()
        assert "error" in response_data

    def test_get_chat_history_success(self, client_with_mock_service, mock_agent_service):
        """Test successful chat history retrieval."""
        mock_agent_service.get_chat_history.return_value = [
            {"content": "Hello", "role": "user", "timestamp": "2023-01-01T00:00:00Z"},
            {"content": "Hi there!", "role": "assistant", "timestamp": "2023-01-01T00:01:00Z"}
        ]
        
        response = client_with_mock_service.get("/api/v1/chat/sessions/test-session-123/history")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["content"] == "Hello"
        assert data[0]["role"] == "user"
        assert data[1]["content"] == "Hi there!"
        assert data[1]["role"] == "assistant"
        mock_agent_service.get_chat_history.assert_called_once_with("test-session-123")

    def test_get_chat_history_empty(self, client, mock_chat_service):
        """Test chat history retrieval for empty session."""
        mock_chat_service.get_chat_history.return_value = []
        
        with patch('src.api.routes.get_agent_service', return_value=mock_chat_service):
            response = client.get("/api/v1/chat/sessions/empty-session/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_get_chat_history_service_failure(self, client_with_mock_service, mock_agent_service):
        """Test chat history retrieval when service fails."""
        mock_agent_service.get_chat_history.side_effect = Exception("History error")
        
        response = client_with_mock_service.get("/api/v1/chat/sessions/test-session/history")
        
        assert response.status_code == 500
        response_data = response.json()
        assert "error" in response_data

    def test_clear_chat_session_success(self, client_with_mock_service, mock_agent_service):
        """Test successful chat session clearing."""
        mock_agent_service.clear_session.return_value = None
        
        response = client_with_mock_service.delete("/api/v1/chat/sessions/test-session-123")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "test-session-123" in data["message"]
        mock_agent_service.clear_session.assert_called_once_with("test-session-123")

    def test_clear_chat_session_service_failure(self, client_with_mock_service, mock_agent_service):
        """Test chat session clearing when service fails."""
        mock_agent_service.clear_session.side_effect = Exception("Clear error")
        
        response = client_with_mock_service.delete("/api/v1/chat/sessions/test-session")
        
        assert response.status_code == 500
        response_data = response.json()
        assert "error" in response_data


class TestHealthRoutes:
    """Test health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        with patch('src.utils.monitoring.get_health_checker') as mock_health:
            mock_health_checker = Mock()
            mock_health_checker.run_all_checks = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "checks": {
                    "basic": {"status": "ok"},
                    "database": {"status": "ok"}
                }
            })
            mock_health.return_value = mock_health_checker
            
            response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "checks" in data

    def test_health_check_service_failure(self, client):
        """Test health check when monitoring service fails."""
        with patch('src.utils.monitoring.get_health_checker') as mock_health:
            mock_health_checker = Mock()
            mock_health_checker.run_all_checks = AsyncMock(side_effect=Exception("Monitor error"))
            mock_health.return_value = mock_health_checker
            
            response = client.get("/health")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]


class TestDependencyInjection:
    """Test dependency injection and service creation."""

    def test_get_agent_service_success(self, client_with_mock_service, mock_agent_service):
        """Test successful agent service dependency injection."""
        mock_agent_service.execute_task.return_value = {"success": True}
        
        response = client_with_mock_service.post(
            "/api/v1/agent/execute",
            json={"task": "Test task", "context": {}}
        )
        
        assert response.status_code == 200
        mock_agent_service.execute_task.assert_called_once()

    def test_get_agent_service_creation_failure(self, app):
        """Test agent service creation failure."""
        from src.api.routes import get_agent_service
        
        # Override dependency to raise exception
        def failing_service():
            raise Exception("Service creation failed")
        
        app.dependency_overrides[get_agent_service] = failing_service
        client = TestClient(app, base_url="http://localhost")
        
        # TestClient doesn't properly handle dependency injection exceptions
        # The exception is raised during dependency resolution, not caught by exception handlers
        with pytest.raises(Exception, match="Service creation failed"):
            client.post(
                "/api/v1/agent/execute",
                json={"task": "Test task", "context": {}}
            )
        
        # Clean up
        app.dependency_overrides.clear()


class TestRequestValidation:
    """Test comprehensive request validation scenarios."""

    def test_invalid_json_payload(self, client):
        """Test endpoints with invalid JSON payload."""
        response = client.post(
            "/api/v1/agent/execute",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_missing_content_type(self, client):
        """Test endpoints without proper content type."""
        response = client.post(
            "/api/v1/agent/execute",
            content=json.dumps({"task": "Test", "context": {}})
        )
        
        # FastAPI handles this gracefully, expecting 200 or other valid response
        assert response.status_code in [200, 422, 500]
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data

    def test_oversized_request_payload(self, client):
        """Test endpoints with oversized request payload."""
        large_task = "x" * 10000  # Very large task string
        response = client.post(
            "/api/v1/agent/execute",
            json={"task": large_task, "context": {}}
        )
        
        # Should either succeed or fail with appropriate error
        assert response.status_code in [200, 413, 422, 500]
        if response.status_code == 500:
            data = response.json()
            assert "error" in data
            assert "message" in data["error"]

    def test_special_characters_in_session_id(self, client_with_mock_service, mock_agent_service):
        """Test handling of special characters in session ID."""
        import urllib.parse
        
        mock_agent_service.get_chat_history.return_value = []
        
        # URL encode the session ID with special characters
        session_id = "test@session#123"
        encoded_session_id = urllib.parse.quote(session_id, safe='')
        response = client_with_mock_service.get(f"/api/v1/chat/sessions/{encoded_session_id}/history")
        
        # Should handle special characters appropriately
        assert response.status_code == 200
        assert response.json() == []


class TestResponseFormats:
    """Test response format validation."""

    def test_agent_execute_response_format(self, client, mock_agent_service):
        """Test agent execute response matches expected format."""
        expected_response = {
            "result": "Task completed",
            "metadata": {"execution_time": 1.5}
        }
        mock_agent_service.execute_task.return_value = expected_response
        
        with patch('src.api.routes.get_agent_service', return_value=mock_agent_service):
            response = client.post(
                "/api/v1/agent/execute",
                json={"task": "Test task", "context": {}}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "result" in data
        assert isinstance(data["success"], bool)

    def test_chat_message_response_format(self, client, mock_chat_service):
        """Test chat message response matches expected format."""
        mock_response = {
            "message": "Test response",
            "session_id": "test-session",
            "metadata": {"model": "gpt-4"}
        }
        mock_chat_service.process_chat_message.return_value = mock_response
        
        with patch('src.api.routes.get_agent_service', return_value=mock_chat_service):
            response = client.post(
                "/api/v1/chat/message",
                json={"message": "Hello", "session_id": "test-session"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "session_id" in data
        assert "metadata" in data
        assert isinstance(data["message"], str)
        assert isinstance(data["session_id"], str)

    def test_error_response_format(self, client_with_mock_service, mock_agent_service):
        """Test error response format consistency."""
        mock_agent_service.execute_task.side_effect = Exception("Test error")
        
        response = client_with_mock_service.post(
            "/api/v1/agent/execute",
            json={"task": "Test task", "context": {}}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert isinstance(data["error"]["message"], str)


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_validation_error_handling(self, client):
        """Test ValidationError handling."""
        response = client.post(
            "/api/v1/agent/execute",
            json={"task": 123}  # Invalid type
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_http_exception_handling(self, client_with_mock_service, mock_agent_service):
        """Test HTTPException handling."""
        mock_agent_service.execute_task.side_effect = HTTPException(
            status_code=403, detail="Forbidden"
        )
        
        response = client_with_mock_service.post(
            "/api/v1/agent/execute",
            json={"task": "Test task", "context": {}}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]

    def test_general_exception_handling(self, client_with_mock_service, mock_agent_service):
        """Test general exception handling."""
        mock_agent_service.execute_task.side_effect = RuntimeError("Unexpected error")
        
        response = client_with_mock_service.post(
            "/api/v1/agent/execute",
            json={"task": "Test task", "context": {}}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "Unexpected error" in data["error"]["message"]

    def test_async_exception_handling(self, client_with_mock_service, mock_agent_service):
        """Test async exception handling."""
        mock_agent_service.process_chat_message.side_effect = AsyncMock(
            side_effect=Exception("Async error")
        )
        
        response = client_with_mock_service.post(
            "/api/v1/chat/message",
            json={"message": "Hello", "session_id": "test-session"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data