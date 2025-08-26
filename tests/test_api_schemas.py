"""Tests for API schemas and validation."""

import pytest
from pydantic import ValidationError
from datetime import datetime

from src.api.schemas import (
  ChatMessage,
  ChatRequest,
  ChatResponse,
  AgentRequest,
  AgentResponse,
  HealthResponse,
  ErrorResponse,
  AgentConfig,
  SessionInfo,
  AgentStatus,
)


class TestChatMessage:
  """Test ChatMessage schema."""

  def test_valid_chat_message(self):
    """Test creating a valid chat message."""
    message = ChatMessage(
      content="Hello, how are you?", role="user", timestamp=datetime.now()
    )
    assert message.content == "Hello, how are you?"
    assert message.role == "user"
    assert isinstance(message.timestamp, datetime)

  def test_chat_message_with_metadata(self):
    """Test chat message with minimal fields."""
    message = ChatMessage(
      content="I'm doing well, thank you!",
      role="assistant",
    )
    assert message.content == "I'm doing well, thank you!"
    assert message.role == "assistant"
    assert message.timestamp is None

  def test_different_roles(self):
    """Test chat message with different roles."""
    for role in ["user", "assistant", "system", "custom_role"]:
      message = ChatMessage(content="Hello", role=role)
      assert message.role == role
      assert message.content == "Hello"

  def test_empty_content_allowed(self):
    """Test chat message with empty content is allowed."""
    message = ChatMessage(content="", role="user")
    assert message.content == ""
    assert message.role == "user"

  def test_whitespace_content_allowed(self):
    """Test chat message with whitespace-only content is allowed."""
    message = ChatMessage(content="   \n\t   ", role="user")
    assert message.content == "   \n\t   "
    assert message.role == "user"


class TestChatRequest:
  """Test ChatRequest schema."""

  def test_valid_chat_request(self):
    """Test creating a valid chat request."""
    request = ChatRequest(message="Hello", session_id="session-123")
    assert request.message == "Hello"
    assert request.session_id == "session-123"
    assert request.context is None  # Default None

  def test_chat_request_with_context(self):
    """Test chat request with context."""
    context = {"previous_topic": "weather", "user_preference": "brief"}
    request = ChatRequest(message="Hello", session_id="session-123", context=context)
    assert request.context == context

  def test_missing_required_fields(self):
    """Test chat request with missing required fields."""
    with pytest.raises(ValidationError):
      ChatRequest(session_id="session-123")  # Missing message

    with pytest.raises(ValidationError):
      ChatRequest(message="Hello")  # Missing session_id


class TestChatResponse:
  """Test ChatResponse schema."""

  def test_valid_chat_response(self):
    """Test creating a valid chat response."""
    response = ChatResponse(
      message="Hello! How can I help you?", session_id="session-123"
    )
    assert response.message == "Hello! How can I help you?"
    assert response.session_id == "session-123"
    assert response.metadata is None

  def test_chat_response_with_metadata(self):
    """Test chat response with metadata."""
    metadata = {"model": "gpt-4", "tokens_used": 150}
    response = ChatResponse(
      message="Response",
      session_id="session-123",
      metadata=metadata,
    )
    assert response.metadata == metadata

  def test_missing_required_fields(self):
    """Test chat response with missing required fields."""
    with pytest.raises(ValidationError):
      ChatResponse(session_id="session-123")  # Missing message

    with pytest.raises(ValidationError):
      ChatResponse(
        message="Hello"
      )  # Missing session_id


class TestAgentRequest:
  """Test AgentRequest schema."""

  def test_valid_agent_request(self):
    """Test creating a valid agent request."""
    request = AgentRequest(task="Create a new file")
    assert request.task == "Create a new file"
    assert request.context is None
    assert request.config is None

  def test_agent_request_with_config(self):
    """Test agent request with configuration."""
    config = {"model_name": "gpt-4", "temperature": 0.7, "timeout": 300}
    request = AgentRequest(
      task="Complex task", config=config
    )
    assert request.config == config

  def test_empty_task_allowed(self):
    """Test agent request with empty task is allowed."""
    request = AgentRequest(task="")
    assert request.task == ""
    assert request.context is None
    assert request.config is None


class TestAgentResponse:
  """Test AgentResponse schema."""

  def test_valid_agent_response(self):
    """Test creating a valid agent response."""
    response = AgentResponse(
      success=True,
      result={"output": "Task completed successfully"},
      message="Task completed successfully",
      execution_time=1.5,
    )
    assert response.success is True
    assert response.result == {"output": "Task completed successfully"}
    assert response.message == "Task completed successfully"
    assert response.execution_time == 1.5

  def test_agent_response_with_actions(self):
    """Test agent response with minimal required fields."""
    response = AgentResponse(
      success=False,
      message="Task failed",
    )
    assert response.success is False
    assert response.message == "Task failed"
    assert response.result is None
    assert response.execution_time is None

  def test_invalid_status(self):
    """Test agent response with missing required fields."""
    with pytest.raises(ValidationError):
      AgentResponse(
        result={"output": "Task completed"}
      )


class TestAgentConfig:
  """Test AgentConfig schema."""

  def test_valid_agent_config(self):
    """Test creating a valid agent config."""
    config = AgentConfig(
      model_name="gpt-4", temperature=0.7, max_tokens=1000, timeout=30.0, streaming=True
    )
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.timeout == 30.0
    assert config.streaming is True

  def test_agent_config_defaults(self):
    """Test agent config with default values."""
    config = AgentConfig()
    assert config.model_name == "gpt-4"
    assert config.temperature == 0.7
    assert config.max_tokens == 1000
    assert config.timeout == 30
    assert config.streaming is False

  def test_invalid_temperature(self):
    """Test agent config with invalid temperature."""
    with pytest.raises(ValidationError):
      AgentConfig(temperature=3.0)

  def test_invalid_timeout(self):
    """Test agent config with invalid timeout."""
    with pytest.raises(ValidationError):
      AgentConfig(timeout=-1.0)


class TestHealthResponse:
  """Test HealthResponse schema."""

  def test_valid_health_response(self):
    """Test creating a valid health response."""
    response = HealthResponse(
      status="healthy", version="1.0.0", service="editor-agent", timestamp=datetime.now()
    )
    assert response.status == "healthy"
    assert response.version == "1.0.0"
    assert response.service == "editor-agent"
    assert isinstance(response.timestamp, datetime)

  def test_invalid_status(self):
    """Test health response with missing required fields."""
    with pytest.raises(ValidationError):
      HealthResponse(status="healthy")


class TestErrorResponse:
  """Test ErrorResponse schema."""

  def test_valid_error_response(self):
    """Test creating a valid error response."""
    response = ErrorResponse(
      error="ValidationError",
      detail="Invalid input data",
      code="400",
      timestamp=datetime.now(),
    )
    assert response.error == "ValidationError"
    assert response.detail == "Invalid input data"
    assert response.code == "400"
    assert isinstance(response.timestamp, datetime)

  def test_error_response_with_minimal_fields(self):
    """Test error response with minimal fields."""
    response = ErrorResponse(error="NotFound")
    assert response.error == "NotFound"
    assert response.detail is None
    assert response.code is None
    assert isinstance(response.timestamp, datetime)


class TestSessionInfo:
  """Test SessionInfo schema."""

  def test_valid_session_info(self):
    """Test creating valid session info."""
    info = SessionInfo(
      session_id="test-session-123",
      created_at=datetime.now(),
      last_activity=datetime.now(),
      message_count=5,
    )
    assert info.session_id == "test-session-123"
    assert info.message_count == 5
    assert isinstance(info.created_at, datetime)
    assert isinstance(info.last_activity, datetime)

  def test_session_info_with_metadata(self):
    """Test session info with metadata."""
    metadata = {"user_id": "123", "client": "web"}
    info = SessionInfo(
      session_id="test-session",
      created_at=datetime.now(),
      last_activity=datetime.now(),
      message_count=0,
      metadata=metadata,
    )
    assert info.metadata == metadata


class TestAgentStatus:
  """Test AgentStatus schema."""

  def test_valid_agent_status(self):
    """Test creating a valid agent status."""
    status = AgentStatus(
      status="running",
      active_sessions=5,
      total_requests=100,
      uptime=3600.0,
      last_activity=datetime.now(),
      memory_usage={"used": 512.5, "total": 1024.0},
    )
    assert status.status == "running"
    assert status.active_sessions == 5
    assert status.total_requests == 100
    assert status.uptime == 3600.0
    assert isinstance(status.last_activity, datetime)
    assert status.memory_usage == {"used": 512.5, "total": 1024.0}

  def test_agent_status_completed(self):
    """Test agent status with minimal fields."""
    status = AgentStatus(
      status="idle",
      active_sessions=0,
      total_requests=50,
      uptime=1800.0,
    )
    assert status.status == "idle"
    assert status.active_sessions == 0
    assert status.total_requests == 50
    assert status.uptime == 1800.0
    assert status.last_activity is None
    assert status.memory_usage is None

  def test_invalid_progress(self):
    """Test agent status with invalid active_sessions."""
    with pytest.raises(ValidationError):
      AgentStatus(
        status="running", active_sessions=-1, total_requests=100
      )

    with pytest.raises(ValidationError):
      AgentStatus(
        status="running", active_sessions=5, total_requests=-10
      )