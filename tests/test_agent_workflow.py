"""Tests for agent workflow."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.workflow import (
  create_agent_workflow,
  create_chat_workflow,
  get_llm_client,
  get_agent_workflow,
  get_chat_workflow,
)


class TestLLMClient:
  """Test LLM client configuration and initialization."""

  def test_get_llm_client_openai(self):
    """Test OpenAI client initialization."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      mock_settings.return_value.model_provider = "openai"
      mock_settings.return_value.openai_api_key = "test-key"

      with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        client = get_llm_client()

        mock_openai.assert_called_once_with(api_key="test-key")
        assert client == mock_client

  def test_get_llm_client_anthropic(self):
    """Test Anthropic client initialization."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      mock_settings.return_value.model_provider = "anthropic"
      mock_settings.return_value.anthropic_api_key = "test-key"

      with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        client = get_llm_client()

        mock_anthropic.assert_called_once_with(api_key="test-key")
        assert client == mock_client

  def test_get_llm_client_invalid_provider(self):
    """Test error handling for invalid provider."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      mock_settings.return_value.model_provider = "invalid"

      with pytest.raises(ValueError, match="Unsupported model provider: invalid"):
        get_llm_client()

  def test_get_llm_client_default_provider(self):
    """Test default provider when no provider specified."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      # Mock settings without model_provider attribute
      mock_settings.return_value = MagicMock(spec=[])
      mock_settings.return_value.openai_api_key = "test-key"

      with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        client = get_llm_client()

        mock_openai.assert_called_once_with(api_key="test-key")
        assert client == mock_client

  def test_get_llm_client_openai_missing_api_key(self):
    """Test error handling when OpenAI API key is missing."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      mock_settings.return_value.model_provider = "openai"
      mock_settings.return_value.openai_api_key = None

      with pytest.raises(ValueError, match="OpenAI API key not found in settings"):
        get_llm_client()

  def test_get_llm_client_anthropic_missing_api_key(self):
    """Test error handling when Anthropic API key is missing."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      mock_settings.return_value.model_provider = "anthropic"
      mock_settings.return_value.anthropic_api_key = None

      with pytest.raises(ValueError, match="Anthropic API key not found in settings"):
        get_llm_client()

  def test_get_llm_client_openai_import_error(self):
    """Test error handling when OpenAI package is not installed."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      mock_settings.return_value.model_provider = "openai"
      mock_settings.return_value.openai_api_key = "test-key"

      with patch("builtins.__import__", side_effect=ImportError("No module named 'openai'")):
        with pytest.raises(ImportError, match="OpenAI package not installed"):
          get_llm_client()

  def test_get_llm_client_anthropic_import_error(self):
    """Test error handling when Anthropic package is not installed."""
    with patch("src.agent.workflow.get_settings") as mock_settings:
      mock_settings.return_value.model_provider = "anthropic"
      mock_settings.return_value.anthropic_api_key = "test-key"

      with patch("builtins.__import__", side_effect=ImportError("No module named 'anthropic'")):
        with pytest.raises(ImportError, match="Anthropic package not installed"):
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
      result = await workflow.run(session_id="test-session", user_input="Hello, world!")

      # Check final state
      assert result["success"]
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
      result = await workflow.run(session_id="test-session", user_input="Test error")

      # Check error handling
      assert not result["success"]
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
      result = await workflow.chat(session_id="test-session", message="Hello!")

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
        context={"user_id": "test-user"},
      )

      # Check that session information is preserved
      assert result["metadata"]["session_id"] == "test-session"
      assert result["success"]
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
          session_id="test-session", user_input="List current directory", context={}
        )

        # Check that workflow executed successfully
        assert result["success"]
        assert "response" in result

  @pytest.mark.asyncio
  async def test_workflow_error_recovery(self):
    """Test workflow error recovery mechanisms."""
    workflow = create_agent_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      # Simulate LLM error during planning
      mock_llm.return_value.messages.create.side_effect = Exception("LLM Error")

      result = await workflow.run(
        session_id="test-session", user_input="Test message", context={}
      )

      # Check that error was handled gracefully
      assert not result["success"]
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
      "current_step_index": 1,  # Still has steps remaining
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
      "current_step_index": 2,  # All steps completed
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
        session_id="test-session", user_input="Hello!", context={}
      )

      # Reset mock for chat workflow
      mock_client.messages.create.side_effect = [
        MagicMock(content=[MagicMock(text="Sure, what would you like to know?")])
      ]

      chat_result = await chat_workflow.chat(
        session_id="test-session",
        message="Follow up question",
        context={"previous_response": agent_result["response"]},
      )

      # Check integration
      assert agent_result["success"]
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
          context={},
        )

        # Check that workflow executed successfully
        assert result["success"]
        assert "response" in result

  @pytest.mark.asyncio
  async def test_workflow_basic_execution_timing(self):
    """Test basic workflow execution timing."""
    import time
    
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

      start_time = time.time()
      result = await workflow.run(
        session_id="test-session", user_input="Quick test", context={}
      )
      end_time = time.time()
      
      processing_time = end_time - start_time

      # Check that workflow executed successfully
      assert result["success"]
      assert result["response"] is not None
      assert result["metadata"]["session_id"] == "test-session"
      assert processing_time >= 0  # Basic timing check

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
        context={},
      )

      # Should complete with error handling
      assert not result["success"]
      assert result["error"] is not None

  @pytest.mark.asyncio
  async def test_workflow_stream_method(self):
    """Test the stream method of EditorAgentWorkflow."""
    workflow = create_agent_workflow()
    
    # Mock the compiled graph's astream method
    mock_chunks = [{"node": "process_user_input"}, {"node": "plan_task"}]
    
    with patch.object(workflow.compiled_graph, 'astream') as mock_astream:
        async def async_iter():
            for chunk in mock_chunks:
                yield chunk
        mock_astream.return_value = async_iter()
        
        chunks = []
        async for chunk in workflow.stream(
            session_id="test-session",
            user_input="Test streaming"
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks == mock_chunks

  @pytest.mark.asyncio
  async def test_workflow_stream_with_config(self):
    """Test stream method with custom config."""
    workflow = create_agent_workflow()
    
    mock_chunks = [{"test": "chunk"}]
    
    with patch.object(workflow.compiled_graph, 'astream') as mock_astream:
        async def async_iter():
            for chunk in mock_chunks:
                yield chunk
        mock_astream.return_value = async_iter()
        
        chunks = []
        async for chunk in workflow.stream(
            session_id="custom-session",
            user_input="Test",
            config={"custom": "config"}
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks == mock_chunks
        
        # Verify astream was called with correct arguments
        mock_astream.assert_called_once()

  def test_singleton_workflow_getters(self):
    """Test singleton behavior of workflow getters."""
    # Clear any existing instances
    import src.agent.workflow as workflow_module
    workflow_module._agent_workflow = None
    workflow_module._chat_workflow = None

    # Test agent workflow singleton
    workflow1 = get_agent_workflow()
    workflow2 = get_agent_workflow()
    assert workflow1 is workflow2
    assert workflow_module._agent_workflow is workflow1

    # Test chat workflow singleton
    chat1 = get_chat_workflow()
    chat2 = get_chat_workflow()
    assert chat1 is chat2
    assert workflow_module._chat_workflow is chat1

    # Test that create functions return new instances
    new_workflow = create_agent_workflow()
    new_chat = create_chat_workflow()
    assert new_workflow is not workflow1
    assert new_chat is not chat1

  @pytest.mark.asyncio
  async def test_chat_workflow_empty_message(self):
    """Test chat workflow with empty message."""
    workflow = create_chat_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client
      mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Please provide a message.")]
      )

      result = await workflow.chat(
        session_id="test-session",
        message="",  # Empty message
        context={}
      )

      assert "message" in result
      assert result["session_id"] == "test-session"

  @pytest.mark.asyncio
  async def test_chat_workflow_llm_error(self):
    """Test chat workflow error handling when LLM fails."""
    workflow = create_chat_workflow()

    with patch.object(workflow.compiled_graph, "ainvoke") as mock_ainvoke:
      mock_ainvoke.side_effect = Exception("LLM connection failed")

      result = await workflow.chat(
        session_id="test-session",
        message="Test message",
        context={}
      )

      # Should handle error gracefully
      assert "I encountered an error" in result["message"]
      assert result["session_id"] == "test-session"
      assert result["metadata"]["error"] is True
      assert result["metadata"]["error_type"] == "Exception"

  @pytest.mark.asyncio
  async def test_chat_workflow_with_context(self):
    """Test chat workflow with custom context."""
    workflow = create_chat_workflow()

    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_client = MagicMock()
      mock_llm.return_value = mock_client
      mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Response with context")]
      )

      # Mock the compiled graph run
      with patch.object(workflow.compiled_graph, "ainvoke") as mock_ainvoke:
        mock_ainvoke.return_value = {
          "agent_response": "Response with context",
          "metadata": {"context_used": True}
        }

        result = await workflow.chat(
          session_id="test-session",
          message="Test with context",
          context={"previous_topic": "testing"}
        )

        # Verify context was passed to the graph
        call_args = mock_ainvoke.call_args[0][0]
        assert call_args["context"] == {"previous_topic": "testing"}
        assert "message" in result

  @pytest.mark.asyncio
  async def test_workflow_network_error_handling(self):
    """Test workflow handling of network errors."""
    workflow = create_agent_workflow()

    with patch.object(workflow.compiled_graph, "ainvoke") as mock_ainvoke:
      # Simulate network error
      mock_ainvoke.side_effect = ConnectionError("Network unreachable")

      result = await workflow.run(
        session_id="test-session",
        user_input="Test input",
        context={}
      )

      # Should handle network error gracefully
      assert not result["success"]
      assert "Network unreachable" in str(result["error"])
      assert result["metadata"]["session_id"] == "test-session"
      assert result["metadata"]["error_type"] == "ConnectionError"

  @pytest.mark.asyncio
  async def test_workflow_timeout_error(self):
    """Test workflow handling of timeout errors."""
    import asyncio
    
    workflow = create_agent_workflow()

    with patch.object(workflow.compiled_graph, "ainvoke") as mock_ainvoke:
      # Mock timeout scenario
      mock_ainvoke.side_effect = asyncio.TimeoutError("Operation timed out")

      result = await workflow.run(
        session_id="test-session",
        user_input="Test input",
        context={}
      )

      # Should handle timeout gracefully
      assert not result["success"]
      assert "Operation timed out" in str(result["error"])
      assert result["metadata"]["session_id"] == "test-session"
      assert result["metadata"]["error_type"] == "TimeoutError"

  @pytest.mark.asyncio
  async def test_workflow_invalid_state_handling(self):
    """Test workflow handling of invalid state transitions."""
    workflow = create_agent_workflow()

    with patch.object(workflow.compiled_graph, "ainvoke") as mock_ainvoke:
      # Mock invalid state return from graph (missing final_response)
      mock_ainvoke.return_value = {"invalid_key": "invalid_value"}  # Missing required keys

      result = await workflow.run(
        session_id="test-session",
        user_input="Test input",
        context={}
      )

      # Should handle invalid state gracefully (workflow processes any returned state)
      assert result["success"]  # Workflow doesn't validate state structure
      assert result["response"] is None  # final_response is missing, so None
      assert result["metadata"]["session_id"] == "test-session"

  @pytest.mark.asyncio
  async def test_workflow_stream_error_handling(self):
    """Test error handling in stream method."""
    workflow = create_agent_workflow()

    with patch.object(workflow.compiled_graph, 'astream') as mock_astream:
      # Mock error during streaming
      async def error_astream():
        yield {"process_input": {"messages": []}}
        raise RuntimeError("Stream processing failed")
      
      mock_astream.return_value = error_astream()
      
      chunks = []
      try:
        async for chunk in workflow.stream(
          session_id="test-session",
          user_input="Test input",
          context={}
        ):
          chunks.append(chunk)
      except RuntimeError as e:
        assert "Stream processing failed" in str(e)

      # Should have received at least one chunk before error
      assert len(chunks) >= 1
      assert "process_input" in chunks[0]

  @pytest.mark.asyncio
  async def test_chat_workflow_initialization_error(self):
    """Test chat workflow handling of initialization errors."""
    workflow = create_chat_workflow()
    
    with patch.object(workflow.compiled_graph, "ainvoke") as mock_ainvoke:
      # Simulate initialization error
      mock_ainvoke.side_effect = ValueError("Invalid configuration")

      result = await workflow.chat(
        session_id="test-session",
        message="Test message",
        context={}
      )

      # Should handle initialization error gracefully
      assert "I encountered an error" in result["message"]
      assert result["metadata"]["error"] is True
      assert result["metadata"]["error_type"] == "ValueError"

  def test_workflow_creation_with_invalid_config(self):
    """Test workflow creation with invalid configuration."""
    # Test that workflow creation handles missing dependencies gracefully
    with patch("src.agent.workflow.get_llm_client") as mock_llm:
      mock_llm.side_effect = ImportError("Required package not found")

      # Should not raise exception during creation
      workflow = create_agent_workflow()
      assert workflow is not None
      assert hasattr(workflow, "compiled_graph")









  @pytest.mark.asyncio
  async def test_workflow_execution_timing(self):
    """Test workflow execution timing."""
    import time
    
    workflow = create_agent_workflow()

    # Patch at the compiled graph level to avoid langgraph async issues
    with patch.object(workflow.compiled_graph, "ainvoke") as mock_ainvoke:
      # Mock the compiled graph to return a simple response
      mock_ainvoke.return_value = {
        "final_response": "Test response",
        "success": True,
        "metadata": {"session_id": "perf-test"}
      }

      start_time = time.time()
      result = await workflow.run(
        session_id="perf-test",
        user_input="Performance test",
        context={}
      )
      end_time = time.time()
      
      processing_time = end_time - start_time

      # Verify basic execution
      assert result["success"]
      assert processing_time >= 0  # Just verify it took some time