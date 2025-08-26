"""Agent state definitions for LangGraph workflows."""

from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
  """Main agent state for LangGraph workflows."""

  # Core conversation state
  messages: List[BaseMessage]
  session_id: str

  # Task and context information
  current_task: Optional[str]
  task_context: Optional[Dict[str, Any]]
  user_input: Optional[str]

  # Agent execution state
  current_step: str
  step_count: int
  max_steps: int

  # Results and outputs
  final_response: Optional[str]
  intermediate_results: List[Dict[str, Any]]
  error_message: Optional[str]

  # Configuration and metadata
  agent_config: Dict[str, Any]
  metadata: Dict[str, Any]
  timestamp: datetime

  # Tool and action state
  available_tools: List[str]
  tool_results: Dict[str, Any]
  next_action: Optional[str]

  # Memory and context
  conversation_history: List[Dict[str, Any]]
  working_memory: Dict[str, Any]
  long_term_memory: Optional[Dict[str, Any]]


class ChatState(TypedDict):
  """Simplified state for chat interactions."""

  messages: List[BaseMessage]
  session_id: str
  user_message: str
  agent_response: Optional[str]
  context: Optional[Dict[str, Any]]
  metadata: Dict[str, Any]


class TaskState(TypedDict):
  """State for specific task execution."""

  task_id: str
  task_description: str
  task_type: str
  status: str  # pending, running, completed, failed

  # Task execution details
  steps: List[Dict[str, Any]]
  current_step_index: int
  results: Dict[str, Any]

  # Context and configuration
  context: Dict[str, Any]
  config: Dict[str, Any]

  # Timing and metadata
  started_at: Optional[datetime]
  completed_at: Optional[datetime]
  execution_time: Optional[float]

  # Error handling
  errors: List[str]
  retry_count: int
  max_retries: int


class ToolState(TypedDict):
  """State for tool execution within workflows."""

  tool_name: str
  tool_input: Dict[str, Any]
  tool_output: Optional[Any]
  execution_status: str  # pending, running, completed, failed
  error_message: Optional[str]
  execution_time: Optional[float]


class WorkflowState(TypedDict):
  """State for complex multi-step workflows."""

  workflow_id: str
  workflow_name: str
  current_node: str

  # Workflow execution
  nodes_completed: List[str]
  nodes_remaining: List[str]
  workflow_status: str  # running, completed, failed, paused

  # Data flow
  node_outputs: Dict[str, Any]
  shared_context: Dict[str, Any]

  # Control flow
  next_node: Optional[str]
  conditional_branches: Dict[str, Any]
  loop_state: Optional[Dict[str, Any]]

  # Metadata
  started_at: datetime
  last_updated: datetime
  metadata: Dict[str, Any]


def create_initial_agent_state(
  session_id: str, task: Optional[str] = None, config: Optional[Dict[str, Any]] = None
) -> AgentState:
  """Create an initial agent state."""
  return AgentState(
    messages=[],
    session_id=session_id,
    current_task=task,
    task_context=None,
    user_input=None,
    current_step="start",
    step_count=0,
    max_steps=config.get("max_steps", 50) if config else 50,
    final_response=None,
    intermediate_results=[],
    error_message=None,
    agent_config=config or {},
    metadata={},
    timestamp=datetime.now(UTC),
    available_tools=[],
    tool_results={},
    next_action=None,
    conversation_history=[],
    working_memory={},
    long_term_memory=None,
  )


def create_chat_state(
  session_id: str, user_message: str, context: Optional[Dict[str, Any]] = None
) -> ChatState:
  """Create a chat state for simple interactions."""
  return ChatState(
    messages=[],
    session_id=session_id,
    user_message=user_message,
    agent_response=None,
    context=context,
    metadata={"created_at": datetime.now(UTC)},
  )


def create_task_state(
  task_id: str,
  task_description: str,
  task_type: str = "general",
  config: Optional[Dict[str, Any]] = None,
) -> TaskState:
  """Create a task state for specific task execution."""
  return TaskState(
    task_id=task_id,
    task_description=task_description,
    task_type=task_type,
    status="pending",
    steps=[],
    current_step_index=0,
    results={},
    context={},
    config=config or {},
    started_at=None,
    completed_at=None,
    execution_time=None,
    errors=[],
    retry_count=0,
    max_retries=config.get("max_retries", 3) if config else 3,
  )