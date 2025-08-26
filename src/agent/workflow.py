"""LangGraph workflow definitions for the editor agent."""

from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.state import AgentState, ChatState, create_initial_agent_state
from src.agent.nodes import (
  process_user_input,
  plan_task,
  execute_action,
  generate_response,
  handle_error,
  should_continue,
  finalize_response,
)
from src.config.settings import get_settings


def get_llm_client():
  """Get LLM client based on configuration."""
  settings = get_settings()

  if hasattr(settings, "model_provider"):
    provider = settings.model_provider
  else:
    # Default to openai if no provider specified
    provider = "openai"

  if provider == "openai":
    try:
      from openai import OpenAI

      api_key = getattr(settings, "openai_api_key", None)
      if not api_key:
        raise ValueError("OpenAI API key not found in settings")
      return OpenAI(api_key=api_key)
    except ImportError:
      raise ImportError(
        "OpenAI package not installed. Install with: pip install openai"
      )

  elif provider == "anthropic":
    try:
      from anthropic import Anthropic

      api_key = getattr(settings, "anthropic_api_key", None)
      if not api_key:
        raise ValueError("Anthropic API key not found in settings")
      return Anthropic(api_key=api_key)
    except ImportError:
      raise ImportError(
        "Anthropic package not installed. Install with: pip install anthropic"
      )

  else:
    raise ValueError(f"Unsupported model provider: {provider}")


class EditorAgentWorkflow:
  """Main workflow for the editor agent using LangGraph."""

  def __init__(self):
    self.settings = get_settings()
    self.graph = self._build_graph()
    self.compiled_graph = self.graph.compile()

  def _build_graph(self) -> StateGraph:
    """Build the LangGraph workflow."""
    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("plan_task", plan_task)
    workflow.add_node("execute_action", execute_action)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("finalize", finalize_response)

    # Set entry point
    workflow.set_entry_point("process_input")

    # Add edges
    workflow.add_edge("process_input", "plan_task")
    workflow.add_edge("plan_task", "execute_action")
    workflow.add_edge("handle_error", "finalize")
    workflow.add_edge("finalize", END)

    # Add conditional edges
    workflow.add_conditional_edges(
      "execute_action",
      should_continue,
      {
        "continue": "execute_action",
        "respond": "generate_response",
        "error": "handle_error",
        "end": "finalize",
      },
    )

    workflow.add_conditional_edges(
      "generate_response",
      lambda state: "end" if state.get("final_response") else "execute_action",
      {"end": "finalize", "execute_action": "execute_action"},
    )

    return workflow

  async def run(
    self,
    session_id: str,
    user_input: str,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Run the agent workflow."""
    # Create initial state
    initial_state = create_initial_agent_state(
      session_id=session_id, task=user_input, config=config
    )

    # Add user message to state
    initial_state["messages"] = [HumanMessage(content=user_input)]
    initial_state["user_input"] = user_input
    initial_state["task_context"] = context or {}

    # Run the workflow
    try:
      result = await self.compiled_graph.ainvoke(
        initial_state, config=RunnableConfig(configurable={"session_id": session_id})
      )

      # Check if there was an error during workflow execution
      if result.get("error_message"):
        return {
          "success": False,
          "error": result.get("error_message"),
          "metadata": {
            "session_id": session_id,
            "step_count": result.get("step_count", 0),
            "execution_time": result.get("execution_time"),
            "intermediate_results": result.get("intermediate_results", []),
          },
        }

      return {
        "success": True,
        "response": result.get("final_response"),
        "metadata": {
          "session_id": session_id,
          "step_count": result.get("step_count", 0),
          "execution_time": result.get("execution_time"),
          "intermediate_results": result.get("intermediate_results", []),
        },
      }
    except Exception as e:
      return {
        "success": False,
        "error": str(e),
        "metadata": {"session_id": session_id, "error_type": type(e).__name__},
      }

  async def stream(
    self,
    session_id: str,
    user_input: str,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
  ):
    """Stream the agent workflow execution."""
    # Create initial state
    initial_state = create_initial_agent_state(
      session_id=session_id, task=user_input, config=config
    )

    # Add user message to state
    initial_state["messages"] = [HumanMessage(content=user_input)]
    initial_state["user_input"] = user_input
    initial_state["task_context"] = context or {}

    # Stream the workflow
    async for chunk in self.compiled_graph.astream(
      initial_state, config=RunnableConfig(configurable={"session_id": session_id})
    ):
      yield chunk


class SimpleChatWorkflow:
  """Simplified workflow for basic chat interactions."""

  def __init__(self):
    self.settings = get_settings()
    self.graph = self._build_chat_graph()
    self.compiled_graph = self.graph.compile()

  def _build_chat_graph(self) -> StateGraph:
    """Build a simple chat workflow."""
    from src.agent.nodes import process_chat_message, generate_chat_response

    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("process_message", process_chat_message)
    workflow.add_node("generate_response", generate_chat_response)

    # Set entry point
    workflow.set_entry_point("process_message")

    # Add edges
    workflow.add_edge("process_message", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow

  async def chat(
    self, session_id: str, message: str, context: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Process a chat message."""
    from src.agent.state import create_chat_state

    # Create chat state
    chat_state = create_chat_state(
      session_id=session_id, user_message=message, context=context
    )

    # Add user message
    chat_state["messages"] = [HumanMessage(content=message)]

    try:
      result = await self.compiled_graph.ainvoke(
        chat_state, config=RunnableConfig(configurable={"session_id": session_id})
      )

      return {
        "message": result.get("agent_response"),
        "session_id": session_id,
        "metadata": result.get("metadata", {}),
      }
    except Exception as e:
      return {
        "message": f"Sorry, I encountered an error: {str(e)}",
        "session_id": session_id,
        "metadata": {"error": True, "error_type": type(e).__name__},
      }


# Global workflow instances
_agent_workflow: Optional[EditorAgentWorkflow] = None
_chat_workflow: Optional[SimpleChatWorkflow] = None


def get_agent_workflow() -> EditorAgentWorkflow:
  """Get the global agent workflow instance."""
  global _agent_workflow
  if _agent_workflow is None:
    _agent_workflow = EditorAgentWorkflow()
  return _agent_workflow


def get_chat_workflow() -> SimpleChatWorkflow:
  """Get the global chat workflow instance."""
  global _chat_workflow
  if _chat_workflow is None:
    _chat_workflow = SimpleChatWorkflow()
  return _chat_workflow


def create_agent_workflow() -> EditorAgentWorkflow:
  """Create a new agent workflow instance."""
  return EditorAgentWorkflow()


def create_chat_workflow() -> SimpleChatWorkflow:
  """Create a new chat workflow instance."""
  return SimpleChatWorkflow()
