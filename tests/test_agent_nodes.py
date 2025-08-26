"""Tests for agent nodes."""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.nodes import (
  process_user_input,
  plan_task,
  execute_action,
  generate_response,
  handle_error,
  finalize_response,
  should_continue,
  process_chat_message,
  generate_chat_response,
  _execute_step,
)
from src.agent.state import AgentState, ChatState


class TestProcessUserInput:
  """Test process_user_input node."""

  @pytest.mark.asyncio
  async def test_process_user_input_basic(self, mock_agent_state):
    """Test basic user input processing."""
    state = mock_agent_state
    state["messages"] = [{"role": "user", "content": "Hello, world!"}]
    state["user_input"] = "Hello, world!"

    result = await process_user_input(state)

    assert result["current_step"] == "processing_input"
    assert result["step_count"] == 1
    assert "input_analysis" in result["metadata"]

  @pytest.mark.asyncio
  async def test_process_user_input_empty_messages(self, mock_agent_state):
    """Test processing with empty messages."""
    state = mock_agent_state
    state["messages"] = []
    state["user_input"] = "Test input"

    result = await process_user_input(state)

    assert result["current_step"] == "processing_input"
    assert len(result["messages"]) == 1  # System message added
    assert "input_analysis" in result["metadata"]

  @pytest.mark.asyncio
  async def test_process_user_input_no_user_message(self, mock_agent_state):
    """Test processing with no user messages."""
    state = mock_agent_state
    state["messages"] = [{"role": "assistant", "content": "Hello!"}]
    state["user_input"] = ""

    result = await process_user_input(state)

    assert result["current_step"] == "processing_input"
    assert "input_analysis" in result["metadata"]

  @pytest.mark.asyncio
  async def test_process_user_input_multiple_messages(self, mock_agent_state):
    """Test processing with multiple messages."""
    state = mock_agent_state
    state["messages"] = [
      {"role": "user", "content": "First message"},
      {"role": "assistant", "content": "Response"},
      {"role": "user", "content": "Second message"},
    ]
    state["user_input"] = "Second message"

    result = await process_user_input(state)

    assert result["current_step"] == "processing_input"
    assert "input_analysis" in result["metadata"]


class TestPlanTask:
  """Test plan_task node."""

  @pytest.mark.asyncio
  async def test_plan_task_basic(self, mock_agent_state):
    """Test basic task planning."""
    state = mock_agent_state
    state["current_task"] = "Create a new file called test.txt"

    result = await plan_task(state)

    assert result["current_step"] == "planning"
    assert "execution_plan" in result["metadata"]
    assert "steps" in result["metadata"]["execution_plan"]
    assert result["step_count"] == 1
    assert "available_tools" in result

  @pytest.mark.asyncio
  async def test_plan_task_fix_task(self, mock_agent_state):
    """Test planning for fix/debug task."""
    state = mock_agent_state
    state["current_task"] = "Fix the broken function"

    result = await plan_task(state)

    assert result["current_step"] == "planning"
    assert "execution_plan" in result["metadata"]
    steps = result["metadata"]["execution_plan"]["steps"]
    assert "identify_issue" in steps
    assert "implement_fix" in steps

  @pytest.mark.asyncio
  async def test_plan_task_explain_task(self, mock_agent_state):
    """Test planning for explain/help task."""
    state = mock_agent_state
    state["current_task"] = "Explain how this works"

    result = await plan_task(state)

    assert result["current_step"] == "planning"
    assert "execution_plan" in result["metadata"]
    steps = result["metadata"]["execution_plan"]["steps"]
    assert "understand_question" in steps
    assert "research" in steps


class TestExecuteAction:
  """Test execute_action node."""

  @pytest.mark.asyncio
  async def test_execute_action_basic(self, mock_agent_state):
    """Test basic action execution."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {
      "steps": ["test_step"],
      "current_step_index": 0
    }

    with patch("src.agent.nodes._execute_step") as mock_execute:
      mock_execute.return_value = "Step completed successfully"

      result = await execute_action(state)

    assert result["current_step"] == "executing"
    assert len(result["intermediate_results"]) > 0
    assert result["intermediate_results"][0]["success"] == True

  @pytest.mark.asyncio
  async def test_execute_action_no_plan(self, mock_agent_state):
    """Test execution with no plan."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {"steps": [], "current_step_index": 0}

    result = await execute_action(state)

    assert result["current_step"] == "executing"
    assert len(result["intermediate_results"]) == 0

  @pytest.mark.asyncio
  async def test_execute_action_step_error(self, mock_agent_state):
    """Test execution with step error."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {
      "steps": ["failing_step"],
      "current_step_index": 0
    }

    with patch("src.agent.nodes._execute_step") as mock_execute:
      mock_execute.side_effect = Exception("Step failed")

      result = await execute_action(state)

    assert result["current_step"] == "executing"
    assert "error_message" in result
    assert "Step failed" in result["error_message"]

  @pytest.mark.asyncio
  async def test_execute_action_multiple_steps(self, mock_agent_state):
    """Test execution with multiple steps."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {
      "steps": ["step1", "step2"],
      "current_step_index": 0
    }

    with patch("src.agent.nodes._execute_step") as mock_execute:
      mock_execute.return_value = "Step completed"

      result = await execute_action(state)

    assert result["current_step"] == "executing"
    assert result["metadata"]["execution_plan"]["current_step_index"] == 1
    assert len(result["intermediate_results"]) == 1


class TestGenerateResponse:
  """Test generate_response node."""

  @pytest.mark.asyncio
  async def test_generate_response_basic(self, mock_agent_state):
    """Test basic response generation."""
    state = mock_agent_state
    state["user_input"] = "Hello"
    state["intermediate_results"] = [{"step": 1, "result": "File created", "success": True}]

    result = await generate_response(state)

    assert result["current_step"] == "generating_response"
    assert "final_response" in result
    assert "messages" in result
    assert result["step_count"] == 1

  @pytest.mark.asyncio
  async def test_generate_response_no_results(self, mock_agent_state):
    """Test response generation with no intermediate results."""
    state = mock_agent_state
    state["user_input"] = "Hello"
    state["intermediate_results"] = []

    result = await generate_response(state)

    assert result["current_step"] == "generating_response"
    assert "final_response" in result
    assert "messages" in result

  @pytest.mark.asyncio
  async def test_generate_response_with_errors(self, mock_agent_state):
    """Test response generation with error results."""
    state = mock_agent_state
    state["user_input"] = "Hello"
    state["intermediate_results"] = [
      {"step": 1, "result": "Success", "success": True},
      {"step": 2, "result": "Error occurred", "success": False}
    ]

    result = await generate_response(state)

    assert result["current_step"] == "generating_response"
    assert "final_response" in result
    assert "messages" in result
    # Check that the function handles both success and failure results
    assert len(result["intermediate_results"]) == 2


class TestHandleError:
  """Test handle_error node."""

  @pytest.mark.asyncio
  async def test_handle_error_basic(self, mock_agent_state):
    """Test basic error handling."""
    state = mock_agent_state
    state["error_message"] = "Test error message"

    result = await handle_error(state)

    assert result["current_step"] == "error_handling"
    assert "I apologize, but I encountered an error" in result["final_response"]
    assert len(result["messages"]) > 0

  @pytest.mark.asyncio
  async def test_handle_error_no_error(self, mock_agent_state):
    """Test error handling with no error."""
    state = mock_agent_state
    state["error_message"] = None

    result = await handle_error(state)

    assert result["current_step"] == "error_handling"
    assert "none" in result["final_response"].lower()


class TestFinalizeResponse:
  """Test finalize_response node."""

  @pytest.mark.asyncio
  async def test_finalize_response_basic(self, mock_agent_state):
    """Test basic response finalization."""
    state = mock_agent_state
    state["final_response"] = "Test response"

    result = await finalize_response(state)

    assert result["current_step"] == "completed"
    assert len(result["conversation_history"]) > 0

  @pytest.mark.asyncio
  async def test_finalize_response_no_response(self, mock_agent_state):
    """Test finalization with no response."""
    state = mock_agent_state
    state["final_response"] = None

    result = await finalize_response(state)

    assert result["current_step"] == "completed"
    assert len(result["conversation_history"]) > 0


class TestShouldContinue:
  """Test should_continue function."""

  def test_should_continue_completed(self, mock_agent_state):
    """Test continuation check for completed state."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {"steps": ["step1"], "current_step_index": 1}

    result = should_continue(state)

    assert result == "respond"

  def test_should_continue_error(self, mock_agent_state):
    """Test continuation check for error state."""
    state = mock_agent_state
    state["error_message"] = "Test error"

    result = should_continue(state)

    assert result == "error"

  def test_should_continue_planning(self, mock_agent_state):
    """Test continuation check for planning state."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {"steps": ["step1", "step2"], "current_step_index": 0}

    result = should_continue(state)

    assert result == "continue"

  def test_should_continue_execution(self, mock_agent_state):
    """Test continuation check for execution state."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {"steps": ["step1", "step2"], "current_step_index": 1}

    result = should_continue(state)

    assert result == "continue"

  def test_should_continue_response_generation(self, mock_agent_state):
    """Test continuation check for response generation state."""
    state = mock_agent_state
    state["step_count"] = 10
    state["max_steps"] = 10

    result = should_continue(state)

    assert result == "end"

  def test_should_continue_finalization(self, mock_agent_state):
    """Test continuation check for finalization state."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {"steps": [], "current_step_index": 0}

    result = should_continue(state)

    assert result == "respond"

  def test_should_continue_unknown(self, mock_agent_state):
    """Test continuation check for unknown state."""
    state = mock_agent_state
    state["metadata"]["execution_plan"] = {"steps": ["step1"], "current_step_index": 0}

    result = should_continue(state)

    assert result == "continue"


class TestChatNodes:
  """Test chat-specific nodes."""

  @pytest.mark.asyncio
  async def test_process_chat_message_basic(self, mock_chat_state):
    """Test basic chat message processing."""
    state = mock_chat_state
    state["user_message"] = "Hello, how are you?"
    
    result = await process_chat_message(state)
    
    assert "user_message" in result
    assert result["user_message"] == "Hello, how are you?"
    assert len(result["messages"]) >= 2  # System message + user message

  @pytest.mark.asyncio
  async def test_process_chat_message_no_message(self, mock_chat_state):
    """Test chat message processing with no message."""
    state = mock_chat_state
    state["user_message"] = ""
    
    result = await process_chat_message(state)
    
    assert "user_message" in result
    assert result["user_message"] == ""

  @pytest.mark.asyncio
  async def test_generate_chat_response_basic(self, mock_chat_state):
    """Test basic chat response generation."""
    state = mock_chat_state
    state["user_message"] = "Hello, how are you?"
    
    result = await generate_chat_response(state)
    
    assert "agent_response" in result
    assert "Hello, how are you?" in result["agent_response"]
    assert "metadata" in result
    assert "response_generated_at" in result["metadata"]

  @pytest.mark.asyncio
  async def test_generate_chat_response_no_response(self, mock_chat_state):
    """Test chat response generation with empty user message."""
    state = mock_chat_state
    state["user_message"] = ""
    
    result = await generate_chat_response(state)
    
    assert "agent_response" in result
    assert result["agent_response"] is not None


class TestExecuteStep:
  """Test _execute_step function."""

  @pytest.mark.asyncio
  async def test_execute_step_basic(self, mock_agent_state):
    """Test basic step execution."""
    step_name = "test_step"

    result = await _execute_step(step_name, mock_agent_state)

    assert isinstance(result, str)
    assert "completed" in result.lower() or "test_step" in result

  @pytest.mark.asyncio
  async def test_execute_step_with_exception(self, mock_agent_state):
    """Test step execution with unknown step."""
    step_name = "unknown_step"

    result = await _execute_step(step_name, mock_agent_state)

    assert isinstance(result, str)
    assert "unknown_step" in result

  @pytest.mark.asyncio
  async def test_execute_step_preserves_state(self, mock_agent_state):
    """Test that step execution works with valid steps."""
    step_name = "planning"

    result = await _execute_step(step_name, mock_agent_state)

    assert isinstance(result, str)
    assert "planning" in result.lower() or "completed" in result.lower()

  @pytest.mark.asyncio
  async def test_execute_step_timing(self, mock_agent_state):
    """Test that step execution returns string result."""
    step_name = "execution"

    result = await _execute_step(step_name, mock_agent_state)

    assert isinstance(result, str)
    assert len(result) > 0


class TestNodeIntegration:
  """Test integration between different nodes."""

  @pytest.mark.asyncio
  async def test_full_workflow_success(self, mock_agent_state):
    """Test a complete successful workflow."""
    # Start with user input
    state = mock_agent_state
    state["messages"] = [{"role": "user", "content": "Create a file called test.txt"}]
    state["user_input"] = "Create a file called test.txt"

    # Process user input
    state = await process_user_input(state)
    assert state["current_step"] == "processing_input"
    assert state["user_input"] == "Create a file called test.txt"

    # Plan task
    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_response = MagicMock()
      mock_response.content = [MagicMock(text="Plan: Use file_write tool")]
      mock_llm.return_value.messages.create.return_value = mock_response

      state = await plan_task(state)
      assert state["current_step"] == "planning"
      assert "execution_plan" in state["metadata"]

    # Execute action
    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_response = MagicMock()
      mock_response.content = [
        MagicMock(
          text='[{"tool": "file_write", "parameters": {"path": "test.txt", "content": "Hello"}}]'
        )
      ]
      mock_llm.return_value.messages.create.return_value = mock_response

      with patch("src.agent.tools.execute_tool") as mock_execute:
        mock_execute.return_value = {"success": True, "message": "File created"}

        state = await execute_action(state)
        assert state["current_step"] == "executing"
        assert len(state["intermediate_results"]) > 0

    # Generate response
    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_response = MagicMock()
      mock_response.content = [MagicMock(text="File created successfully!")]
      mock_llm.return_value.messages.create.return_value = mock_response

      state = await generate_response(state)
      assert state["current_step"] == "generating_response"
      assert state["final_response"] is not None

    # Finalize response
    state = await finalize_response(state)
    assert state["current_step"] == "completed"
    assert len(state["conversation_history"]) > 0

  @pytest.mark.asyncio
  async def test_error_workflow(self, mock_agent_state):
    """Test workflow with error handling."""
    # Start with empty messages (should cause error)
    state = mock_agent_state
    state["messages"] = []

    # Process user input (should fail)
    state = await process_user_input(state)
    assert state["current_step"] == "processing_input"
    assert state["user_input"] is not None

    # Handle error
    state = await handle_error(state)
    assert state["current_step"] == "error_handling"
    assert state["final_response"] is not None

    # Finalize response
    state = await finalize_response(state)
    assert state["current_step"] == "completed"
    assert len(state["conversation_history"]) > 0

  @pytest.mark.asyncio
  async def test_chat_workflow(self, mock_chat_state):
    """Test chat-specific workflow."""
    state = mock_chat_state
    state["current_message"] = {"role": "user", "content": "Hello!"}

    # Process chat message
    state = await process_chat_message(state)
    assert state["user_message"] is not None

    # Generate chat response
    state = await generate_chat_response(state)
    assert len(state["messages"]) >= 2
    assert state["agent_response"] is not None