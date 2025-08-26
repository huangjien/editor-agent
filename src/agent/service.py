"""Agent service for handling agent operations and chat interactions."""

import uuid
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from src.agent.workflow import create_agent_workflow, create_chat_workflow
from src.agent.state import ChatState
from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.exceptions import EditorAgentException, ValidationError


class AgentService:
  """Service for managing agent operations and chat sessions."""

  def __init__(self):
    """Initialize the agent service."""
    self.settings = get_settings()
    self.logger = get_logger(__name__)
    self.chat_sessions: Dict[str, ChatState] = {}
    self.agent_workflow = None
    self.chat_workflow = None
    self._initialize_workflows()

  def _initialize_workflows(self):
    """Initialize the agent and chat workflows."""
    try:
      self.agent_workflow = create_agent_workflow()
      self.chat_workflow = create_chat_workflow()
      self.logger.info("Agent workflows initialized successfully")
    except Exception as e:
      self.logger.error(f"Failed to initialize workflows: {e}")
      raise EditorAgentException(f"Workflow initialization failed: {e}")

  async def execute_task(
    self,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Execute an agent task.

    Args:
        task: The task description
        context: Optional context for the task
        config: Optional configuration overrides

    Returns:
        Dict containing the task execution result
    """
    if not task or not task.strip():
      raise ValidationError("Task cannot be empty")

    try:
      self.logger.info(f"Executing task: {task[:100]}...")

      # Execute the workflow
      if self.agent_workflow is None:
        raise EditorAgentException("Agent workflow not initialized")

      # Generate a session ID for this task
      session_id = str(uuid.uuid4())

      result = await self.agent_workflow.run(
        session_id=session_id, user_input=task, context=context, config=config
      )

      self.logger.info("Task completed successfully")

      return {
        "task_id": session_id,
        "status": "completed",
        "result": result,
        "actions_taken": [],
        "execution_time": 0,
        "timestamp": datetime.now(UTC).isoformat(),
      }

    except Exception as e:
      self.logger.error(f"Task execution failed: {e}")
      raise EditorAgentException(f"Task execution failed: {e}")

  async def get_status(self) -> Dict[str, Any]:
    """Get the current status of the agent service.

    Returns:
        Dict containing status information
    """
    try:
      return {
        "service_status": "active",
        "workflows_initialized": {
          "agent": self.agent_workflow is not None,
          "chat": self.chat_workflow is not None,
        },
        "active_sessions": len(self.chat_sessions),
        "settings": {
          "max_iterations": self.settings.max_iterations,
          "max_execution_time": self.settings.max_execution_time,
          "default_model": self.settings.default_model,
        },
        "timestamp": datetime.now(UTC).isoformat(),
      }
    except Exception as e:
      self.logger.error(f"Failed to get status: {e}")
      raise EditorAgentException(f"Status retrieval failed: {e}")

  async def process_chat_message(
    self,
    message: str,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Process a chat message.

    Args:
        message: The chat message
        session_id: Optional session ID (will create new if not provided)
        context: Optional context for the message

    Returns:
        Dict containing the chat response
    """
    if not message or not message.strip():
      raise ValidationError("Message cannot be empty")

    # Generate session ID if not provided
    if session_id is None:
      session_id = str(uuid.uuid4())

    try:
      # Get or create chat session
      if session_id not in self.chat_sessions:
        from src.agent.state import create_chat_state

        self.chat_sessions[session_id] = create_chat_state(
          session_id=session_id, user_message=message.strip(), context=context or {}
        )
      else:
        # Update existing session with new message
        chat_state = self.chat_sessions[session_id]
        from langchain_core.messages import HumanMessage

        chat_state["messages"].append(HumanMessage(content=message.strip()))
        chat_state["user_message"] = message.strip()

      chat_state = self.chat_sessions[session_id]

      self.logger.info(f"Processing chat message in session {session_id}")

      # Process with chat workflow
      if self.chat_workflow is None:
        raise EditorAgentException("Chat workflow not initialized")

      result = await self.chat_workflow.chat(
        session_id=session_id, message=message.strip(), context=context
      )

      # Update the session with the result
      if "message" in result:
        chat_state["agent_response"] = result["message"]
        from langchain_core.messages import AIMessage

        chat_state["messages"].append(AIMessage(content=result["message"]))

      return {
        "message": result.get("message", "No response generated"),
        "role": "assistant",
        "session_id": session_id,
        "metadata": result.get("metadata", {}),
      }

    except Exception as e:
      self.logger.error(f"Chat message processing failed: {e}")
      raise EditorAgentException(f"Chat processing failed: {e}")

  async def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
    """Get chat history for a session.

    Args:
        session_id: The session ID

    Returns:
        List of chat messages
    """
    if session_id not in self.chat_sessions:
      return []

    try:
      chat_state = self.chat_sessions[session_id]
      return [
        {
          "content": msg["content"],
          "role": msg["role"],
          "timestamp": msg.get("timestamp", datetime.now(UTC).isoformat()),
        }
        for msg in chat_state.messages
      ]
    except Exception as e:
      self.logger.error(f"Failed to get chat history: {e}")
      raise EditorAgentException(f"Chat history retrieval failed: {e}")

  async def clear_session(self, session_id: str) -> None:
    """Clear a chat session.

    Args:
        session_id: The session ID to clear
    """
    try:
      if session_id in self.chat_sessions:
        del self.chat_sessions[session_id]
        self.logger.info(f"Cleared chat session: {session_id}")
      else:
        self.logger.warning(f"Session not found: {session_id}")
    except Exception as e:
      self.logger.error(f"Failed to clear session: {e}")
      raise EditorAgentException(f"Session clearing failed: {e}")

  async def list_sessions(self) -> List[Dict[str, Any]]:
    """List all active chat sessions.

    Returns:
        List of session information
    """
    try:
      return [
        {
          "session_id": session_id,
          "message_count": len(state.messages),
          "created_at": state.created_at.isoformat()
          if hasattr(state, "created_at")
          else None,
          "last_activity": state.updated_at.isoformat()
          if hasattr(state, "updated_at")
          else None,
        }
        for session_id, state in self.chat_sessions.items()
      ]
    except Exception as e:
      self.logger.error(f"Failed to list sessions: {e}")
      raise EditorAgentException(f"Session listing failed: {e}")


# Global service instance
_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
  """Get the global agent service instance.

  Returns:
      AgentService instance
  """
  global _agent_service
  if _agent_service is None:
    _agent_service = AgentService()
  return _agent_service
