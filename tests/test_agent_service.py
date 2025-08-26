"""Tests for agent service."""

from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agent.service import AgentService, get_agent_service
from src.utils.exceptions import EditorAgentException, ValidationError


class TestAgentServiceInitialization:
  """Test AgentService initialization and workflow setup."""

  def test_init_success(self, test_settings):
    """Test successful AgentService initialization."""
    with (
      patch("src.agent.service.get_settings") as mock_get_settings,
      patch("src.agent.service.get_logger") as mock_get_logger,
      patch("src.agent.service.create_agent_workflow") as mock_create_agent,
      patch("src.agent.service.create_chat_workflow") as mock_create_chat,
    ):
      mock_get_settings.return_value = test_settings
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger
      mock_agent_workflow = MagicMock()
      mock_chat_workflow = MagicMock()
      mock_create_agent.return_value = mock_agent_workflow
      mock_create_chat.return_value = mock_chat_workflow

      service = AgentService()

      assert service.settings == test_settings
      assert service.logger == mock_logger
      assert service.chat_sessions == {}
      assert service.agent_workflow == mock_agent_workflow
      assert service.chat_workflow == mock_chat_workflow
      mock_logger.info.assert_called_with("Agent workflows initialized successfully")

  def test_init_workflow_failure(self, test_settings):
    """Test AgentService initialization with workflow creation failure."""
    with (
      patch("src.agent.service.get_settings") as mock_get_settings,
      patch("src.agent.service.get_logger") as mock_get_logger,
      patch("src.agent.service.create_agent_workflow") as mock_create_agent,
    ):
      mock_get_settings.return_value = test_settings
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger
      mock_create_agent.side_effect = Exception("Workflow creation failed")

      with pytest.raises(EditorAgentException, match="Workflow initialization failed"):
        AgentService()

      mock_logger.error.assert_called_with(
        "Failed to initialize workflows: Workflow creation failed"
      )

  def test_initialize_workflows_success(self, test_settings):
    """Test successful workflow initialization."""
    with (
      patch("src.agent.service.get_settings") as mock_get_settings,
      patch("src.agent.service.get_logger") as mock_get_logger,
      patch("src.agent.service.create_agent_workflow") as mock_create_agent,
      patch("src.agent.service.create_chat_workflow") as mock_create_chat,
    ):
      mock_get_settings.return_value = test_settings
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger
      mock_agent_workflow = MagicMock()
      mock_chat_workflow = MagicMock()
      mock_create_agent.return_value = mock_agent_workflow
      mock_create_chat.return_value = mock_chat_workflow

      service = AgentService()

      assert service.agent_workflow is not None
      assert service.chat_workflow is not None
      mock_create_agent.assert_called_once()
      mock_create_chat.assert_called_once()


class TestAgentServiceExecuteTask:
  """Test AgentService execute_task method."""

  @pytest.fixture
  def mock_service(self, test_settings):
    """Create a mock AgentService for testing."""
    with (
      patch("src.agent.service.get_settings") as mock_get_settings,
      patch("src.agent.service.get_logger") as mock_get_logger,
      patch("src.agent.service.create_agent_workflow") as mock_create_agent,
      patch("src.agent.service.create_chat_workflow") as mock_create_chat,
    ):
      mock_get_settings.return_value = test_settings
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger
      mock_agent_workflow = AsyncMock()
      mock_chat_workflow = AsyncMock()
      mock_create_agent.return_value = mock_agent_workflow
      mock_create_chat.return_value = mock_chat_workflow

      service = AgentService()
      return service

  @pytest.mark.asyncio
  async def test_execute_task_success(self, mock_service):
    """Test successful task execution."""
    task = "Test task"
    context = {"key": "value"}
    config = {"setting": "test"}

    mock_result = {"output": "Task completed", "status": "success"}
    mock_service.agent_workflow.run.return_value = mock_result

    with patch("uuid.uuid4") as mock_uuid:
      mock_uuid.return_value = "test-session-id"

      result = await mock_service.execute_task(task, context, config)

      assert result["task_id"] == "test-session-id"
      assert result["status"] == "completed"
      assert result["result"] == mock_result
      assert "timestamp" in result
      assert "actions_taken" in result
      assert "execution_time" in result

      mock_service.agent_workflow.run.assert_called_once_with(
        session_id="test-session-id", user_input=task, context=context, config=config
      )

  @pytest.mark.asyncio
  async def test_execute_task_empty_task(self, mock_service):
    """Test execute_task with empty task."""
    with pytest.raises(ValidationError, match="Task cannot be empty"):
      await mock_service.execute_task("")

    with pytest.raises(ValidationError, match="Task cannot be empty"):
      await mock_service.execute_task("   ")

    with pytest.raises(ValidationError, match="Task cannot be empty"):
      await mock_service.execute_task(None)

  @pytest.mark.asyncio
  async def test_execute_task_no_workflow(self, mock_service):
    """Test execute_task when agent workflow is not initialized."""
    mock_service.agent_workflow = None

    with pytest.raises(EditorAgentException, match="Agent workflow not initialized"):
      await mock_service.execute_task("Test task")

  @pytest.mark.asyncio
  async def test_execute_task_workflow_error(self, mock_service):
    """Test execute_task when workflow execution fails."""
    task = "Test task"
    mock_service.agent_workflow.run.side_effect = Exception("Workflow failed")

    with pytest.raises(EditorAgentException, match="Task execution failed"):
      await mock_service.execute_task(task)

    mock_service.logger.error.assert_called_with(
      "Task execution failed: Workflow failed"
    )

  @pytest.mark.asyncio
  async def test_execute_task_with_defaults(self, mock_service):
    """Test execute_task with default parameters."""
    task = "Test task"
    mock_result = {"output": "Success"}
    mock_service.agent_workflow.run.return_value = mock_result

    await mock_service.execute_task(task)

    mock_service.agent_workflow.run.assert_called_once()
    call_args = mock_service.agent_workflow.run.call_args
    assert call_args[1]["user_input"] == task
    assert call_args[1]["context"] is None
    assert call_args[1]["config"] is None


class TestAgentServiceGetStatus:
  """Test AgentService get_status method."""

  @pytest.fixture
  def mock_service(self, test_settings):
    """Create a mock AgentService for testing."""
    with (
      patch("src.agent.service.get_settings") as mock_get_settings,
      patch("src.agent.service.get_logger") as mock_get_logger,
      patch("src.agent.service.create_agent_workflow") as mock_create_agent,
      patch("src.agent.service.create_chat_workflow") as mock_create_chat,
    ):
      mock_get_settings.return_value = test_settings
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger
      mock_agent_workflow = MagicMock()
      mock_chat_workflow = MagicMock()
      mock_create_agent.return_value = mock_agent_workflow
      mock_create_chat.return_value = mock_chat_workflow

      service = AgentService()
      return service

  @pytest.mark.asyncio
  async def test_get_status_success(self, mock_service):
    """Test successful status retrieval."""
    # Add some mock sessions
    mock_service.chat_sessions = {"session1": MagicMock(), "session2": MagicMock()}

    status = await mock_service.get_status()

    assert status["service_status"] == "active"
    assert status["workflows_initialized"]["agent"] is True
    assert status["workflows_initialized"]["chat"] is True
    assert status["active_sessions"] == 2
    assert "settings" in status
    assert "timestamp" in status

    settings = status["settings"]
    assert "max_iterations" in settings
    assert "max_execution_time" in settings
    assert "default_model" in settings

  @pytest.mark.asyncio
  async def test_get_status_no_workflows(self, mock_service):
    """Test status when workflows are not initialized."""
    mock_service.agent_workflow = None
    mock_service.chat_workflow = None

    status = await mock_service.get_status()

    assert status["workflows_initialized"]["agent"] is False
    assert status["workflows_initialized"]["chat"] is False

  @pytest.mark.asyncio
  async def test_get_status_error(self, mock_service):
    """Test get_status when an error occurs."""
    mock_service.settings = None  # This will cause an error

    with pytest.raises(EditorAgentException, match="Status retrieval failed"):
      await mock_service.get_status()

    mock_service.logger.error.assert_called()


class TestAgentServiceChatFunctionality:
  """Test AgentService chat-related methods."""

  @pytest.fixture
  def mock_service(self, test_settings):
    """Create a mock AgentService for testing."""
    with (
      patch("src.agent.service.get_settings") as mock_get_settings,
      patch("src.agent.service.get_logger") as mock_get_logger,
      patch("src.agent.service.create_agent_workflow") as mock_create_agent,
      patch("src.agent.service.create_chat_workflow") as mock_create_chat,
    ):
      mock_get_settings.return_value = test_settings
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger
      mock_agent_workflow = AsyncMock()
      mock_chat_workflow = AsyncMock()
      mock_create_agent.return_value = mock_agent_workflow
      mock_create_chat.return_value = mock_chat_workflow

      service = AgentService()
      return service

  @pytest.mark.asyncio
  async def test_process_chat_message_new_session(self, mock_service):
    """Test processing chat message with new session."""
    message = "Hello, how are you?"
    mock_result = {"message": "I'm doing well, thank you!"}
    mock_service.chat_workflow.chat.return_value = mock_result

    with (
      patch("src.agent.state.create_chat_state") as mock_create_state,
      patch("uuid.uuid4") as mock_uuid,
    ):
      mock_uuid.return_value = "test-session-id"
      mock_chat_state = {
        "messages": [],
        "user_message": message,
        "agent_response": None,
      }
      mock_create_state.return_value = mock_chat_state

      result = await mock_service.process_chat_message(message)

      assert result["message"] == "I'm doing well, thank you!"
      assert result["role"] == "assistant"
      assert result["session_id"] == "test-session-id"
      assert "metadata" in result

      mock_create_state.assert_called_once_with(
        session_id="test-session-id", user_message=message, context={}
      )

  @pytest.mark.asyncio
  async def test_process_chat_message_existing_session(self, mock_service):
    """Test processing chat message with existing session."""
    session_id = "existing-session"
    message = "Follow up question"

    # Setup existing session
    mock_chat_state = {
      "messages": [HumanMessage(content="Previous message")],
      "user_message": "Previous message",
      "agent_response": None,
    }
    mock_service.chat_sessions[session_id] = mock_chat_state

    mock_result = {"message": "Follow up response"}
    mock_service.chat_workflow.chat.return_value = mock_result

    result = await mock_service.process_chat_message(message, session_id)

    assert result["session_id"] == session_id
    assert mock_chat_state["user_message"] == message
    assert len(mock_chat_state["messages"]) == 3  # Previous + new human + new AI

  @pytest.mark.asyncio
  async def test_process_chat_message_empty_message(self, mock_service):
    """Test processing empty chat message."""
    with pytest.raises(ValidationError, match="Message cannot be empty"):
      await mock_service.process_chat_message("")

    with pytest.raises(ValidationError, match="Message cannot be empty"):
      await mock_service.process_chat_message("   ")

    with pytest.raises(ValidationError, match="Message cannot be empty"):
      await mock_service.process_chat_message(None)

  @pytest.mark.asyncio
  async def test_process_chat_message_no_workflow(self, mock_service):
    """Test processing chat message when chat workflow is not initialized."""
    mock_service.chat_workflow = None

    with pytest.raises(EditorAgentException, match="Chat workflow not initialized"):
      await mock_service.process_chat_message("Test message")

  @pytest.mark.asyncio
  async def test_process_chat_message_workflow_error(self, mock_service):
    """Test processing chat message when workflow fails."""
    message = "Test message"
    mock_service.chat_workflow.chat.side_effect = Exception("Chat failed")

    with patch("src.agent.state.create_chat_state") as mock_create_state:
      mock_create_state.return_value = {"messages": []}

      with pytest.raises(EditorAgentException, match="Chat processing failed"):
        await mock_service.process_chat_message(message)

  @pytest.mark.asyncio
  async def test_get_chat_history_existing_session(self, mock_service):
    """Test getting chat history for existing session."""
    session_id = "test-session"
    mock_messages = [
      {"content": "Hello", "role": "user", "timestamp": "2024-01-01T00:00:00Z"},
      {
        "content": "Hi there!",
        "role": "assistant",
        "timestamp": "2024-01-01T00:00:01Z",
      },
    ]

    # Create a proper mock ChatState object
    mock_chat_state = MagicMock()
    mock_chat_state.messages = mock_messages
    mock_service.chat_sessions[session_id] = mock_chat_state

    history = await mock_service.get_chat_history(session_id)

    assert len(history) == 2
    assert history[0]["content"] == "Hello"
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"
    assert history[1]["role"] == "assistant"

  @pytest.mark.asyncio
  async def test_get_chat_history_nonexistent_session(self, mock_service):
    """Test getting chat history for non-existent session."""
    history = await mock_service.get_chat_history("nonexistent-session")
    assert history == []

  @pytest.mark.asyncio
  async def test_get_chat_history_error(self, mock_service):
    """Test get_chat_history when an error occurs."""
    session_id = "test-session"
    # Create a mock that will raise an exception when iterating over messages
    mock_chat_state = MagicMock()
    mock_chat_state.messages = MagicMock()
    mock_chat_state.messages.__iter__ = MagicMock(
      side_effect=Exception("Iteration error")
    )
    mock_service.chat_sessions[session_id] = mock_chat_state

    with pytest.raises(EditorAgentException, match="Chat history retrieval failed"):
      await mock_service.get_chat_history(session_id)

  @pytest.mark.asyncio
  async def test_clear_session_existing(self, mock_service):
    """Test clearing an existing session."""
    session_id = "test-session"
    mock_service.chat_sessions[session_id] = MagicMock()

    await mock_service.clear_session(session_id)

    assert session_id not in mock_service.chat_sessions
    mock_service.logger.info.assert_called_with(f"Cleared chat session: {session_id}")

  @pytest.mark.asyncio
  async def test_clear_session_nonexistent(self, mock_service):
    """Test clearing a non-existent session."""
    session_id = "nonexistent-session"

    await mock_service.clear_session(session_id)

    mock_service.logger.warning.assert_called_with(f"Session not found: {session_id}")

  @pytest.mark.asyncio
  async def test_clear_session_error(self, mock_service):
    """Test clear_session when an error occurs."""
    session_id = "test-session"

    # Mock an error during deletion by patching the logger to raise an exception
    mock_service.chat_sessions[session_id] = MagicMock()

    with patch.object(
      mock_service.logger, "info", side_effect=Exception("Logging failed")
    ):
      with pytest.raises(EditorAgentException, match="Session clearing failed"):
        await mock_service.clear_session(session_id)

  @pytest.mark.asyncio
  async def test_list_sessions_with_sessions(self, mock_service):
    """Test listing sessions when sessions exist."""
    # Create mock sessions with different attributes
    session1 = MagicMock()
    session1.messages = [MagicMock(), MagicMock()]
    session1.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    session1.updated_at = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

    session2 = MagicMock()
    session2.messages = [MagicMock()]
    # No created_at/updated_at attributes
    del session2.created_at
    del session2.updated_at

    mock_service.chat_sessions = {"session1": session1, "session2": session2}

    sessions = await mock_service.list_sessions()

    assert len(sessions) == 2

    # Check session1
    session1_info = next(s for s in sessions if s["session_id"] == "session1")
    assert session1_info["message_count"] == 2
    assert session1_info["created_at"] == "2024-01-01T00:00:00+00:00"
    assert session1_info["last_activity"] == "2024-01-01T12:00:00+00:00"

    # Check session2
    session2_info = next(s for s in sessions if s["session_id"] == "session2")
    assert session2_info["message_count"] == 1
    assert session2_info["created_at"] is None
    assert session2_info["last_activity"] is None

  @pytest.mark.asyncio
  async def test_list_sessions_empty(self, mock_service):
    """Test listing sessions when no sessions exist."""
    sessions = await mock_service.list_sessions()
    assert sessions == []

  @pytest.mark.asyncio
  async def test_list_sessions_error(self, mock_service):
    """Test list_sessions when an error occurs."""
    # Mock an error during iteration
    mock_service.chat_sessions = MagicMock()
    mock_service.chat_sessions.items.side_effect = Exception("Iteration failed")

    with pytest.raises(EditorAgentException, match="Session listing failed"):
      await mock_service.list_sessions()


class TestGlobalAgentService:
  """Test global agent service functionality."""

  def test_get_agent_service_singleton(self):
    """Test that get_agent_service returns the same instance."""
    with patch("src.agent.service.AgentService") as mock_agent_service_class:
      mock_instance = MagicMock()
      mock_agent_service_class.return_value = mock_instance

      # Reset the global instance
      import src.agent.service

      src.agent.service._agent_service = None

      # First call should create instance
      service1 = get_agent_service()
      assert service1 == mock_instance
      mock_agent_service_class.assert_called_once()

      # Second call should return same instance
      service2 = get_agent_service()
      assert service2 == mock_instance
      assert service1 is service2
      # Should not create another instance
      mock_agent_service_class.assert_called_once()

  def test_get_agent_service_existing_instance(self):
    """Test get_agent_service when instance already exists."""
    with patch("src.agent.service.AgentService") as mock_agent_service_class:
      mock_instance = MagicMock()

      # Set existing instance
      import src.agent.service

      src.agent.service._agent_service = mock_instance

      service = get_agent_service()
      assert service == mock_instance
      # Should not create new instance
      mock_agent_service_class.assert_not_called()


class TestAgentServiceEdgeCases:
  """Test edge cases and error conditions."""

  @pytest.fixture
  def mock_service(self, test_settings):
    """Create a mock AgentService for testing."""
    with (
      patch("src.agent.service.get_settings") as mock_get_settings,
      patch("src.agent.service.get_logger") as mock_get_logger,
      patch("src.agent.service.create_agent_workflow") as mock_create_agent,
      patch("src.agent.service.create_chat_workflow") as mock_create_chat,
    ):
      mock_get_settings.return_value = test_settings
      mock_logger = MagicMock()
      mock_get_logger.return_value = mock_logger
      mock_agent_workflow = AsyncMock()
      mock_chat_workflow = AsyncMock()
      mock_create_agent.return_value = mock_agent_workflow
      mock_create_chat.return_value = mock_chat_workflow

      service = AgentService()
      return service

  @pytest.mark.asyncio
  async def test_execute_task_very_long_task(self, mock_service):
    """Test execute_task with very long task description."""
    long_task = "A" * 1000  # Very long task
    mock_result = {"output": "Success"}
    mock_service.agent_workflow.run.return_value = mock_result

    result = await mock_service.execute_task(long_task)

    assert result["status"] == "completed"
    # Check that logger was called with truncated task
    mock_service.logger.info.assert_any_call(f"Executing task: {long_task[:100]}...")

  @pytest.mark.asyncio
  async def test_process_chat_message_with_context(self, mock_service):
    """Test processing chat message with custom context."""
    message = "Test message"
    context = {"user_id": "123", "preferences": {"theme": "dark"}}

    mock_result = {"message": "Response"}
    mock_service.chat_workflow.chat.return_value = mock_result

    with patch("src.agent.state.create_chat_state") as mock_create_state:
      mock_create_state.return_value = {"messages": []}

      result = await mock_service.process_chat_message(message, context=context)

      mock_create_state.assert_called_once_with(
        session_id=result["session_id"], user_message=message, context=context
      )

  @pytest.mark.asyncio
  async def test_chat_workflow_no_message_in_result(self, mock_service):
    """Test chat processing when workflow result has no message."""
    message = "Test message"
    mock_result = {"metadata": {"some": "data"}}  # No message field
    mock_service.chat_workflow.chat.return_value = mock_result

    with patch("src.agent.state.create_chat_state") as mock_create_state:
      mock_create_state.return_value = {"messages": []}

      result = await mock_service.process_chat_message(message)

      assert result["message"] == "No response generated"
      assert result["metadata"] == {"some": "data"}

  @pytest.mark.asyncio
  async def test_get_chat_history_message_without_timestamp(self, mock_service):
    """Test getting chat history with messages missing timestamps."""
    session_id = "test-session"
    mock_messages = [
      {"content": "Hello", "role": "user"},  # No timestamp
      {"content": "Hi!", "role": "assistant", "timestamp": "2024-01-01T00:00:00Z"},
    ]

    # Create a proper mock ChatState object
    mock_chat_state = MagicMock()
    mock_chat_state.messages = mock_messages
    mock_service.chat_sessions[session_id] = mock_chat_state

    history = await mock_service.get_chat_history(session_id)

    assert len(history) == 2
    assert "timestamp" in history[0]  # Should have default timestamp
    assert history[1]["timestamp"] == "2024-01-01T00:00:00Z"

  def test_service_state_consistency(self, mock_service):
    """Test that service maintains consistent state."""
    # Verify initial state
    assert isinstance(mock_service.chat_sessions, dict)
    assert len(mock_service.chat_sessions) == 0

    # Add a session manually
    session_id = "test-session"
    mock_service.chat_sessions[session_id] = MagicMock()

    assert len(mock_service.chat_sessions) == 1
    assert session_id in mock_service.chat_sessions

  @pytest.mark.asyncio
  async def test_concurrent_session_operations(self, mock_service):
    """Test concurrent operations on chat sessions."""
    import asyncio

    # Create multiple sessions concurrently
    async def create_session(session_id):
      mock_service.chat_sessions[session_id] = MagicMock()

    session_ids = [f"session-{i}" for i in range(5)]
    await asyncio.gather(*[create_session(sid) for sid in session_ids])

    assert len(mock_service.chat_sessions) == 5
    for session_id in session_ids:
      assert session_id in mock_service.chat_sessions

  @pytest.mark.asyncio
  async def test_memory_cleanup_after_clear_session(self, mock_service):
    """Test that memory is properly cleaned up after clearing sessions."""
    # Create multiple sessions
    for i in range(10):
      mock_service.chat_sessions[f"session-{i}"] = MagicMock()

    assert len(mock_service.chat_sessions) == 10

    # Clear all sessions
    for session_id in list(mock_service.chat_sessions.keys()):
      await mock_service.clear_session(session_id)

    assert len(mock_service.chat_sessions) == 0