"""Tests for agent state management."""

from datetime import datetime

from src.agent.state import (
  ToolState,
  WorkflowState,
  create_initial_agent_state,
  create_chat_state,
  create_task_state,
)


class TestAgentState:
  """Test AgentState functionality."""

  def test_create_agent_state(self):
    """Test creating initial agent state."""
    state = create_initial_agent_state("test-session", "Test task")

    assert state["session_id"] == "test-session"
    assert state["current_task"] == "Test task"
    assert state["user_input"] is None  # user_input is None by default
    assert state["current_step"] == "start"
    assert state["step_count"] == 0
    assert state["max_steps"] == 50
    assert state["messages"] == []
    assert state["intermediate_results"] == []
    assert state["error_message"] is None
    assert isinstance(state["metadata"], dict)
    assert isinstance(state["agent_config"], dict)

  def test_agent_state_with_context(self):
    """Test creating agent state with context."""
    context = {"user_id": "123", "workspace": "/tmp/test"}
    state = create_initial_agent_state("test-session", "Test input")
    state["task_context"] = context

    assert state["task_context"] == context

  def test_agent_state_modification(self):
    """Test modifying agent state."""
    state = create_initial_agent_state("test-session", "Test input")

    # Modify state
    state["current_step"] = "planning"
    state["step_count"] = 1
    state["user_input"] = "Modified input"

    assert state["current_step"] == "planning"
    assert state["step_count"] == 1
    assert state["user_input"] == "Modified input"

  def test_agent_state_messages(self):
    """Test managing messages in agent state."""
    state = create_initial_agent_state("test-session", "Test input")

    # Add messages
    state["messages"].append(
      {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()}
    )
    state["messages"].append(
      {
        "role": "assistant",
        "content": "Hi there!",
        "timestamp": datetime.now().isoformat(),
      }
    )

    assert len(state["messages"]) == 2
    assert state["messages"][0]["role"] == "user"
    assert state["messages"][1]["role"] == "assistant"

  def test_agent_state_actions(self):
    """Test agent state actions tracking."""
    state = create_initial_agent_state("test-session", "Test input")

    # Add intermediate results
    state["intermediate_results"].append(
      {"type": "file_read", "path": "/test/file.txt", "result": "File content"}
    )
    state["intermediate_results"].append(
      {"type": "command_execute", "command": "ls -la", "result": "Directory listing"}
    )

    assert len(state["intermediate_results"]) == 2
    assert state["intermediate_results"][0]["type"] == "file_read"
    assert state["intermediate_results"][1]["type"] == "command_execute"


class TestChatState:
  """Test ChatState functionality."""

  def test_create_chat_state(self):
    """Test creating a new chat state."""
    state = create_chat_state("chat-session", "Hello, world!")

    assert state["session_id"] == "chat-session"
    assert state["user_message"] == "Hello, world!"
    assert state["messages"] == []
    assert state["context"] is None
    assert "created_at" in state["metadata"]

  def test_chat_state_with_context(self):
    """Test creating chat state with context."""
    context = {"user_preferences": {"theme": "dark"}}
    state = create_chat_state("chat-session", "Hello!", context)

    assert state["context"] == context

  def test_chat_state_messages(self):
    """Test managing messages in chat state."""
    state = create_chat_state("chat-session", "What's the weather?")

    # Add messages
    state["messages"].append(
      {
        "role": "user",
        "content": "What's the weather?",
        "timestamp": datetime.now().isoformat(),
      }
    )
    state["messages"].append(
      {
        "role": "assistant",
        "content": "I can't check the weather directly.",
        "timestamp": datetime.now().isoformat(),
      }
    )

    assert len(state["messages"]) == 2
    assert state["messages"][0]["content"] == "What's the weather?"
    assert state["messages"][1]["content"] == "I can't check the weather directly."

  def test_chat_state_metadata(self):
    """Test managing metadata in chat state."""
    state = create_chat_state("chat-session", "Test message")

    # Add metadata
    state["metadata"]["created_at"] = datetime.now().isoformat()
    state["metadata"]["model"] = "gpt-4"
    state["metadata"]["token_count"] = 150

    assert "created_at" in state["metadata"]
    assert state["metadata"]["model"] == "gpt-4"
    assert state["metadata"]["token_count"] == 150


class TestTaskState:
  """Test TaskState functionality."""

  def test_create_task_state(self):
    """Test creating a new task state."""
    state = create_task_state("task-123", "Create a file")

    assert state["task_id"] == "task-123"
    assert state["task_description"] == "Create a file"
    assert state["status"] == "pending"
    assert state["steps"] == []
    assert state["current_step_index"] == 0
    assert state["results"] == {}
    assert state["errors"] == []
    assert state["config"] == {}

  def test_task_state_with_steps(self):
    """Test task state with predefined steps."""
    state = create_task_state("task-123", "Complex task")

    # Manually add steps to test the functionality
    steps = ["analyze", "plan", "execute", "verify"]
    state["steps"] = [{"name": step, "status": "pending"} for step in steps]

    assert len(state["steps"]) == 4
    assert state["steps"][0]["name"] == "analyze"

  def test_task_state_progression(self):
    """Test task state progression through steps."""
    state = create_task_state("task-123", "Multi-step task")

    # Add steps manually
    steps = ["step1", "step2", "step3"]
    state["steps"] = [{"name": step, "status": "pending"} for step in steps]

    # Progress through steps
    state["status"] = "running"
    state["current_step_index"] = 1

    assert state["status"] == "running"
    assert state["current_step_index"] == 1

    # Complete task
    state["status"] = "completed"
    state["current_step_index"] = len(steps)
    state["results"]["final_result"] = "Task completed successfully"

    assert state["status"] == "completed"
    assert state["results"]["final_result"] == "Task completed successfully"

  def test_task_state_error_handling(self):
    """Test task state error handling."""
    state = create_task_state("task-123", "Failing task")

    # Set error state
    state["status"] = "failed"
    state["errors"].append("File not found")
    state["config"]["error_code"] = "FILE_NOT_FOUND"

    assert state["status"] == "failed"
    assert "File not found" in state["errors"]
    assert state["config"]["error_code"] == "FILE_NOT_FOUND"


class TestToolState:
  """Test ToolState functionality."""

  def test_tool_state_structure(self):
    """Test tool state structure."""
    # ToolState is a TypedDict, so we test its structure
    tool_state: ToolState = {
      "tool_name": "file_read",
      "parameters": {"path": "/test/file.txt"},
      "result": "File content",
      "error": None,
      "execution_time": 0.5,
      "metadata": {"file_size": 1024},
    }

    assert tool_state["tool_name"] == "file_read"
    assert tool_state["parameters"]["path"] == "/test/file.txt"
    assert tool_state["result"] == "File content"
    assert tool_state["error"] is None
    assert tool_state["execution_time"] == 0.5
    assert tool_state["metadata"]["file_size"] == 1024

  def test_tool_state_error(self):
    """Test tool state with error."""
    tool_state: ToolState = {
      "tool_name": "file_write",
      "parameters": {"path": "/readonly/file.txt", "content": "test"},
      "result": None,
      "error": "Permission denied",
      "execution_time": 0.1,
      "metadata": {"error_code": "PERMISSION_DENIED"},
    }

    assert tool_state["error"] == "Permission denied"
    assert tool_state["result"] is None
    assert tool_state["metadata"]["error_code"] == "PERMISSION_DENIED"


class TestWorkflowState:
  """Test WorkflowState functionality."""

  def test_workflow_state_structure(self):
    """Test workflow state structure."""
    workflow_state: WorkflowState = {
      "workflow_id": "workflow-123",
      "status": "running",
      "current_node": "planning",
      "nodes_completed": ["input", "validation"],
      "context": {"user_id": "123"},
      "metadata": {"started_at": datetime.now().isoformat()},
    }

    assert workflow_state["workflow_id"] == "workflow-123"
    assert workflow_state["status"] == "running"
    assert workflow_state["current_node"] == "planning"
    assert len(workflow_state["nodes_completed"]) == 2
    assert "user_id" in workflow_state["context"]
    assert "started_at" in workflow_state["metadata"]

  def test_workflow_state_progression(self):
    """Test workflow state progression."""
    workflow_state: WorkflowState = {
      "workflow_id": "workflow-123",
      "status": "pending",
      "current_node": "start",
      "nodes_completed": [],
      "context": {},
      "metadata": {},
    }

    # Progress workflow
    workflow_state["status"] = "running"
    workflow_state["current_node"] = "processing"
    workflow_state["nodes_completed"].append("start")

    assert workflow_state["status"] == "running"
    assert workflow_state["current_node"] == "processing"
    assert "start" in workflow_state["nodes_completed"]

    # Complete workflow
    workflow_state["status"] = "completed"
    workflow_state["current_node"] = "end"
    workflow_state["nodes_completed"].extend(["processing", "end"])

    assert workflow_state["status"] == "completed"
    assert len(workflow_state["nodes_completed"]) == 3


class TestStateIntegration:
  """Test integration between different state types."""

  def test_agent_state_with_task_states(self):
    """Test agent state containing task states."""
    agent_state = create_initial_agent_state("agent-session", "Complex workflow")

    # Add task states to agent metadata
    task1 = create_task_state("task-1", "Read file")
    task2 = create_task_state("task-2", "Process data")

    agent_state["metadata"]["tasks"] = [task1, task2]

    assert len(agent_state["metadata"]["tasks"]) == 2
    assert agent_state["metadata"]["tasks"][0]["task_id"] == "task-1"
    assert agent_state["metadata"]["tasks"][1]["task_id"] == "task-2"

  def test_chat_state_with_agent_context(self):
    """Test chat state with agent context."""
    chat_state = create_chat_state("chat-session", "Hello")

    # Add agent context (initialize context first since it's None by default)
    chat_state["context"] = {}
    chat_state["context"]["agent_session"] = "agent-123"
    chat_state["context"]["capabilities"] = ["file_ops", "code_gen"]

    assert chat_state["context"]["agent_session"] == "agent-123"
    assert "file_ops" in chat_state["context"]["capabilities"]

  def test_state_serialization(self):
    """Test that states can be serialized (important for persistence)."""
    import json

    agent_state = create_initial_agent_state("test-session", "Test input")
    # Convert datetime to string for JSON serialization
    agent_state["timestamp"] = agent_state["timestamp"].isoformat()
    agent_state["metadata"]["custom_timestamp"] = datetime.now().isoformat()

    # Should be serializable to JSON
    serialized = json.dumps(agent_state)
    deserialized = json.loads(serialized)

    assert deserialized["session_id"] == "test-session"
    assert deserialized["current_task"] == "Test input"
    assert "custom_timestamp" in deserialized["metadata"]
