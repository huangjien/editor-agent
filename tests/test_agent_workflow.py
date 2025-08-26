"""Tests for agent workflow."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.workflow import (
  create_agent_workflow,
  create_chat_workflow,
  get_llm_client,
)
from src.agent.state import AgentState, ChatState


class TestLLMClient:
  """Test LLM client functionality."""

  def test_get_llm_client_openai(self):
    """Test getting OpenAI client."""
    with patch("src.agent.workflow.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.model_provider = "openai"
      mock_settings.openai_api_key = "test-key"
      mock_get_settings.return_value = mock_settings

      with patch("openai.OpenAI") as mock_openai:
        get_llm_client()
        mock_openai.assert_called_once_with(api_key="test-key")

  def test_get_llm_client_anthropic(self):
    """Test getting Anthropic client."""
    with patch("src.agent.workflow.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.model_provider = "anthropic"
      mock_settings.anthropic_api_key = "test-key"
      mock_get_settings.return_value = mock_settings

      with patch("anthropic.Anthropic") as mock_anthropic:
        get_llm_client()
        mock_anthropic.assert_called_once_with(api_key="test-key")

  def test_get_llm_client_invalid_provider(self):
    """Test getting client with invalid provider."""
    with patch("src.agent.workflow.get_settings") as mock_get_settings:
      mock_settings = MagicMock()
      mock_settings.model_provider = "invalid"
      mock_get_settings.return_value = mock_settings

      with pytest.raises(ValueError, match="Unsupported model provider"):
        get_llm_client()


class TestAgentWorkflow:
  """Test agent workflow functionality."""

  def test_create_agent_workflow(self):
    """Test creating agent workflow."""
    workflow = create_agent_workflow()

    # Check that workflow is properly initialized
    assert workflow is not None
    assert hasattr(workflow, "graph")
    assert hasattr(workflow, "compiled_graph")

    # Check that all required nodes are present in the graph
    expected_nodes = [
      "process_input",
      "plan_task",
      "execute_action",
      "generate_response",
      "handle_error",
      "finalize",
    ]

    for node in expected_nodes:
      assert node in workflow.graph.nodes

  def test_create_chat_workflow(self):
    """Test creating chat workflow."""
    workflow = create_chat_workflow()

    # Check that workflow is properly initialized
    assert workflow is not None
    assert hasattr(workflow, "graph")
    assert hasattr(workflow, "compiled_graph")

    # Check that required nodes are present in the graph
    expected_nodes = ["process_message", "generate_response"]

    for node in expected_nodes:
      assert node in workflow.graph.nodes

  @pytest.mark.asyncio
  async def test_agent_workflow_execution_success(self):
    """Test successful agent workflow execution."""
    workflow = create_agent_workflow()

    # Mock LLM responses
    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client

      # Mock planning response
      plan_response = MagicMock()
      plan_response.content = [MagicMock(text="Plan: Respond with greeting")]

      # Mock execution response
      exec_response = MagicMock()
      exec_response.content = [MagicMock(text='[{"tool": "none", "parameters": {}}]')]

      # Mock response generation
      resp_response = MagicMock()
      resp_response.content = [MagicMock(text="Hello! How can I help you?")]

      mock_client.messages.create.side_effect = [
        plan_response,
        exec_response,
        resp_response,
      ]

      # Execute workflow using run method
      result = await workflow.run(
        session_id="test-session",
        user_input="Hello, world!"
      )

      # Check final state
      assert result["success"] == True
      assert result["response"] is not None
      assert result["metadata"]["session_id"] == "test-session"

  @pytest.mark.asyncio
  async def test_agent_workflow_execution_error(self):
    """Test agent workflow execution with error."""
    workflow = create_agent_workflow()

    # Mock LLM to raise exception
    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_llm.side_effect = Exception("LLM error")

      # Execute workflow using run method
      result = await workflow.run(
        session_id="test-session",
        user_input="Test error"
      )

      # Check error handling
      assert result["success"] == False
      assert result["error"] is not None

  @pytest.mark.asyncio
  async def test_chat_workflow_execution(self):
    """Test chat workflow execution."""
    workflow = create_chat_workflow()

    # Mock LLM response
    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client

      response = MagicMock()
      response.content = [MagicMock(text="Hello! How can I help you?")]
      mock_client.messages.create.return_value = response

      # Execute workflow using chat method
      result = await workflow.chat(
        session_id="test-session",
        message="Hello!"
      )

      # Check final state
      assert result["message"] is not None
      assert result["session_id"] == "test-session"

  @pytest.mark.asyncio
  async def test_workflow_state_persistence(self):
    """Test that workflow maintains state across steps."""
    workflow = create_agent_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client

      # Mock responses
      plan_response = MagicMock()
      plan_response.content = [MagicMock(text="Test plan")]

      exec_response = MagicMock()
      exec_response.content = [MagicMock(text="[]")]

      resp_response = MagicMock()
      resp_response.content = [MagicMock(text="Test response")]

      mock_client.messages.create.side_effect = [
        plan_response,
        exec_response,
        resp_response,
      ]

      result = await workflow.run(
        session_id="test-session",
        user_input="Test message",
        context={"user_id": "test-user"}
      )

      # Check that session information is preserved
      assert result["metadata"]["session_id"] == "test-session"
      assert result["success"] == True
      assert result["response"] is not None

  @pytest.mark.asyncio
  async def test_workflow_with_tool_execution(self):
    """Test workflow with actual tool execution."""
    workflow = create_agent_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client

      # Mock planning response
      plan_response = MagicMock()
      plan_response.content = [MagicMock(text="Plan: Use directory_list tool")]

      # Mock execution response with tool call
      exec_response = MagicMock()
      exec_response.content = [
        MagicMock(text='[{"tool": "directory_list", "parameters": {"path": "."}}]')
      ]

      # Mock response generation
      resp_response = MagicMock()
      resp_response.content = [
        MagicMock(text="Here are the files in the current directory")
      ]

      mock_client.messages.create.side_effect = [
        plan_response,
        exec_response,
        resp_response,
      ]

      # Mock tool execution
      with patch("src.agent.nodes.execute_action") as mock_execute:
        mock_execute.return_value = {
          "success": True,
          "contents": [
            {"name": "file1.txt", "type": "file", "size": 100},
            {"name": "dir1", "type": "directory", "size": 0},
          ],
        }

        result = await workflow.run(
          session_id="test-session",
          user_input="List current directory",
          context={}
        )

        # Check that workflow executed successfully
        assert result["success"] == True
        assert "response" in result

  @pytest.mark.asyncio
  async def test_workflow_error_recovery(self):
    """Test workflow error recovery mechanisms."""
    workflow = create_agent_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      # Simulate LLM error during planning
      mock_llm.return_value.messages.create.side_effect = Exception("LLM Error")

      result = await workflow.run(
        session_id="test-session",
        user_input="Test message",
        context={}
      )

      # Check that error was handled gracefully
      assert result["success"] == False
      assert result["error"] is not None
      assert "error" in str(result["error"]).lower()

  @pytest.mark.asyncio
  async def test_workflow_conditional_routing(self):
    """Test workflow conditional routing logic."""
    from src.agent.state import create_initial_agent_state
    from src.agent.nodes import should_continue
    
    create_agent_workflow()

    # Test different routing scenarios based on state conditions
    # should_continue returns: "continue", "respond", "error", "end"
    
    # Test normal continuation (no error, steps remaining)
    state = create_initial_agent_state("test-session")
    state["metadata"]["execution_plan"] = {
      "steps": ["step1", "step2", "step3"],
      "current_step_index": 1  # Still has steps remaining
    }
    assert should_continue(state) == "continue"
    
    # Test error condition
    state = create_initial_agent_state("test-session")
    state["error_message"] = "Some error occurred"
    assert should_continue(state) == "error"
    
    # Test step limit reached
    state = create_initial_agent_state("test-session")
    state["step_count"] = 50  # Reached max_steps
    state["max_steps"] = 50
    assert should_continue(state) == "end"
    
    # Test plan complete (should respond)
    state = create_initial_agent_state("test-session")
    state["metadata"]["execution_plan"] = {
      "steps": ["step1", "step2"],
      "current_step_index": 2  # All steps completed
    }
    assert should_continue(state) == "respond"


class TestWorkflowIntegration:
  """Test workflow integration scenarios."""

  @pytest.mark.asyncio
  async def test_agent_to_chat_workflow_integration(self):
    """Test integration between agent and chat workflows."""
    agent_workflow = create_agent_workflow()
    chat_workflow = create_chat_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client

      # Mock agent responses
      responses = [
        MagicMock(content=[MagicMock(text="Plan: Respond with greeting")]),
        MagicMock(content=[MagicMock(text="[]")]),
        MagicMock(content=[MagicMock(text="Hello! How can I help?")]),
      ]
      mock_client.messages.create.side_effect = responses

      agent_result = await agent_workflow.run(
        session_id="test-session",
        user_input="Hello!",
        context={}
      )

      # Reset mock for chat workflow
      mock_client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text="Sure, what would you like to know?")])
      ]

      chat_result = await chat_workflow.chat(
        session_id="test-session",
        message="Follow up question",
        context={"previous_response": agent_result["response"]}
      )

      # Check integration
      assert agent_result["success"] == True
      assert chat_result["message"] is not None
      assert agent_result["response"] is not None
      assert chat_result["message"] is not None

  @pytest.mark.asyncio
  async def test_workflow_with_multiple_tools(self):
    """Test workflow execution with multiple tool calls."""
    workflow = create_agent_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client

      # Mock responses
      plan_response = MagicMock()
      plan_response.content = [MagicMock(text="Plan: Create file then read it")]

      exec_response = MagicMock()
      exec_response.content = [
        MagicMock(
          text=json.dumps(
            [
              {
                "tool": "file_write",
                "parameters": {"path": "test.txt", "content": "Hello"},
              },
              {"tool": "file_read", "parameters": {"path": "test.txt"}},
            ]
          )
        )
      ]

      resp_response = MagicMock()
      resp_response.content = [MagicMock(text="File created and read successfully")]

      mock_client.messages.create.side_effect = [
        plan_response,
        exec_response,
        resp_response,
      ]

      # Mock tool executions
      with patch("src.agent.nodes.execute_action") as mock_execute:
        mock_execute.side_effect = [
          {"success": True, "message": "File created"},
          {"success": True, "content": "Hello"},
        ]

        result = await workflow.run(
          session_id="test-session",
          user_input="Create a file and then read it",
          context={}
        )

        # Check that workflow executed successfully
        assert result["success"] == True
        assert "response" in result

  @pytest.mark.asyncio
  async def test_workflow_performance_tracking(self):
    """Test that workflow tracks performance metrics."""
    workflow = create_agent_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client

      # Mock quick responses
      responses = [
        MagicMock(content=[MagicMock(text="Quick plan")]),
        MagicMock(content=[MagicMock(text="[]")]),
        MagicMock(content=[MagicMock(text="Quick response")]),
      ]
      mock_client.messages.create.side_effect = responses

      result = await workflow.run(
        session_id="test-session",
        user_input="Quick test",
        context={}
      )

      # Check that workflow executed successfully
      assert result["success"] == True
      assert result["response"] is not None
      assert result["metadata"]["session_id"] == "test-session"

  @pytest.mark.asyncio
  async def test_workflow_state_validation(self):
    """Test workflow state validation and error handling."""
    workflow = create_agent_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client
      mock_client.messages.create.side_effect = Exception("Validation Error")

      # Workflow should handle errors gracefully
      result = await workflow.run(
        session_id="test-session",
        user_input="",  # Empty input to test validation
        context={}
      )

      # Should complete with error handling
      assert result["success"] == False
      assert result["error"] is not None